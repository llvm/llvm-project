//===----------- MemoryManager.cpp - Target independent memory manager ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functionality for managing target memory.
// It is very expensive to call alloc/free functions of target devices. The
// MemoryManagerTy in this file is to reduce the number of invocations of those
// functions by buffering allocated device memory. In this way, when a memory is
// not used, it will not be freed on the device directly. The buffer is
// organized in a number of buckets for efficient look up. A memory will go to
// corresponding bucket based on its size. When a new memory request comes in,
// it will first check whether there is free memory of same size. If yes,
// returns it directly. Otherwise, allocate one on device.
//
// It also provides a way to opt out the memory manager. Memory
// allocation/deallocation will only be managed if the requested size is less
// than SizeThreshold, which can be configured via an environment variable
// LIBOMPTARGET_MEMORY_MANAGER_THRESHOLD.
//
//===----------------------------------------------------------------------===//

#include "MemoryManager.h"
#include "device.h"
#include "private.h"
#include "rtl.h"

namespace {
constexpr const size_t BucketSize[] = {
    0,       1U << 2, 1U << 3,  1U << 4,  1U << 5,  1U << 6, 1U << 7,
    1U << 8, 1U << 9, 1U << 10, 1U << 11, 1U << 12, 1U << 13};

constexpr const int NumBuckets = sizeof(BucketSize) / sizeof(BucketSize[0]);

/// The threshold to manage memory using memory manager. If the request size is
/// larger than \p SizeThreshold, the allocation will not be managed by the
/// memory manager. This variable can be configured via an env \p
/// LIBOMPTARGET_MEMORY_MANAGER_THRESHOLD. By default, the value is 8KB.
size_t SizeThreshold = 1U << 13;

/// Find the previous number that is power of 2 given a number that is not power
/// of 2.
size_t floorToPowerOfTwo(size_t Num) {
  Num |= Num >> 1;
  Num |= Num >> 2;
  Num |= Num >> 4;
  Num |= Num >> 8;
  Num |= Num >> 16;
  Num |= Num >> 32;
  Num += 1;
  return Num >> 1;
}

/// Find a suitable bucket
int findBucket(size_t Size) {
  const size_t F = floorToPowerOfTwo(Size);

  DP("findBucket: Size %zu is floored to %zu.\n", Size, F);

  int L = 0, H = NumBuckets - 1;
  while (H - L > 1) {
    int M = (L + H) >> 1;
    if (BucketSize[M] == F)
      return M;
    if (BucketSize[M] > F)
      H = M - 1;
    else
      L = M;
  }

  assert(L >= 0 && L < NumBuckets && "L is out of range");

  DP("findBucket: Size %zu goes to bucket %d\n", Size, L);

  return L;
}
} // namespace

MemoryManagerTy::MemoryManagerTy(DeviceTy &Dev, size_t Threshold)
    : FreeLists(NumBuckets), FreeListLocks(NumBuckets), Device(Dev) {
  if (Threshold)
    SizeThreshold = Threshold;
}

MemoryManagerTy::~MemoryManagerTy() {
  // TODO: There is a little issue that target plugin is destroyed before this
  // object, therefore the memory free will not succeed.
  // Deallocate all memory in map
  for (auto Itr = PtrToNodeTable.begin(); Itr != PtrToNodeTable.end(); ++Itr) {
    assert(Itr->second.Ptr && "nullptr in map table");
    deleteOnDevice(Itr->second.Ptr);
  }
}

void *MemoryManagerTy::allocateOnDevice(size_t Size, void *HstPtr) const {
  return Device.RTL->data_alloc(Device.RTLDeviceID, Size, HstPtr);
}

int MemoryManagerTy::deleteOnDevice(void *Ptr) const {
  return Device.RTL->data_delete(Device.RTLDeviceID, Ptr);
}

void *MemoryManagerTy::freeAndAllocate(size_t Size, void *HstPtr) {
  std::vector<void *> RemoveList;

  // Deallocate all memory in FreeList
  for (int I = 0; I < NumBuckets; ++I) {
    FreeListTy &List = FreeLists[I];
    std::lock_guard<std::mutex> Lock(FreeListLocks[I]);
    if (List.empty())
      continue;
    for (const NodeTy &N : List) {
      deleteOnDevice(N.Ptr);
      RemoveList.push_back(N.Ptr);
    }
    FreeLists[I].clear();
  }

  // Remove all nodes in the map table which have been released
  if (!RemoveList.empty()) {
    std::lock_guard<std::mutex> LG(MapTableLock);
    for (void *P : RemoveList)
      PtrToNodeTable.erase(P);
  }

  // Try allocate memory again
  return allocateOnDevice(Size, HstPtr);
}

void *MemoryManagerTy::allocateOrFreeAndAllocateOnDevice(size_t Size,
                                                         void *HstPtr) {
  void *TgtPtr = allocateOnDevice(Size, HstPtr);
  // We cannot get memory from the device. It might be due to OOM. Let's
  // free all memory in FreeLists and try again.
  if (TgtPtr == nullptr) {
    DP("Failed to get memory on device. Free all memory in FreeLists and "
       "try again.\n");
    TgtPtr = freeAndAllocate(Size, HstPtr);
  }

#ifdef OMPTARGET_DEBUG
  if (TgtPtr == nullptr)
    DP("Still cannot get memory on device probably because the device is "
       "OOM.\n");
#endif

  return TgtPtr;
}

void *MemoryManagerTy::allocate(size_t Size, void *HstPtr) {
  // If the size is zero, we will not bother the target device. Just return
  // nullptr directly.
  if (Size == 0)
    return nullptr;

  DP("MemoryManagerTy::allocate: size %zu with host pointer " DPxMOD ".\n",
     Size, DPxPTR(HstPtr));

  // If the size is greater than the threshold, allocate it directly from
  // device.
  if (Size > SizeThreshold) {
    DP("%zu is greater than the threshold %zu. Allocate it directly from "
       "device\n",
       Size, SizeThreshold);
    void *TgtPtr = allocateOrFreeAndAllocateOnDevice(Size, HstPtr);

    DP("Got target pointer " DPxMOD ". Return directly.\n", DPxPTR(TgtPtr));

    return TgtPtr;
  }

  NodeTy *NodePtr = nullptr;

  // Try to get a node from FreeList
  {
    const int B = findBucket(Size);
    FreeListTy &List = FreeLists[B];

    NodeTy TempNode(Size, nullptr);
    std::lock_guard<std::mutex> LG(FreeListLocks[B]);
    FreeListTy::const_iterator Itr = List.find(TempNode);

    if (Itr != List.end()) {
      NodePtr = &Itr->get();
      List.erase(Itr);
    }
  }

#ifdef OMPTARGET_DEBUG
  if (NodePtr != nullptr)
    DP("Find one node " DPxMOD " in the bucket.\n", DPxPTR(NodePtr));
#endif

  // We cannot find a valid node in FreeLists. Let's allocate on device and
  // create a node for it.
  if (NodePtr == nullptr) {
    DP("Cannot find a node in the FreeLists. Allocate on device.\n");
    // Allocate one on device
    void *TgtPtr = allocateOrFreeAndAllocateOnDevice(Size, HstPtr);

    if (TgtPtr == nullptr)
      return nullptr;

    // Create a new node and add it into the map table
    {
      std::lock_guard<std::mutex> Guard(MapTableLock);
      auto Itr = PtrToNodeTable.emplace(TgtPtr, NodeTy(Size, TgtPtr));
      NodePtr = &Itr.first->second;
    }

    DP("Node address " DPxMOD ", target pointer " DPxMOD ", size %zu\n",
       DPxPTR(NodePtr), DPxPTR(TgtPtr), Size);
  }

  assert(NodePtr && "NodePtr should not be nullptr at this point");

  return NodePtr->Ptr;
}

int MemoryManagerTy::free(void *TgtPtr) {
  DP("MemoryManagerTy::free: target memory " DPxMOD ".\n", DPxPTR(TgtPtr));

  NodeTy *P = nullptr;

  // Look it up into the table
  {
    std::lock_guard<std::mutex> G(MapTableLock);
    auto Itr = PtrToNodeTable.find(TgtPtr);

    // We don't remove the node from the map table because the map does not
    // change.
    if (Itr != PtrToNodeTable.end())
      P = &Itr->second;
  }

  // The memory is not managed by the manager
  if (P == nullptr) {
    DP("Cannot find its node. Delete it on device directly.\n");
    return deleteOnDevice(TgtPtr);
  }

  // Insert the node to the free list
  const int B = findBucket(P->Size);

  DP("Found its node " DPxMOD ". Insert it to bucket %d.\n", DPxPTR(P), B);

  {
    std::lock_guard<std::mutex> G(FreeListLocks[B]);
    FreeLists[B].insert(*P);
  }

  return OFFLOAD_SUCCESS;
}
