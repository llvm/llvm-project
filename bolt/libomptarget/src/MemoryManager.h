//===----------- MemoryManager.h - Target independent memory manager ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declarations for target independent memory manager.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPENMP_LIBOMPTARGET_SRC_MEMORYMANAGER_H
#define LLVM_OPENMP_LIBOMPTARGET_SRC_MEMORYMANAGER_H

#include <cassert>
#include <functional>
#include <list>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

// Forward declaration
struct DeviceTy;

class MemoryManagerTy {
  /// A structure stores the meta data of a target pointer
  struct NodeTy {
    /// Memory size
    const size_t Size;
    /// Target pointer
    void *Ptr;

    /// Constructor
    NodeTy(size_t Size, void *Ptr) : Size(Size), Ptr(Ptr) {}
  };

  /// To make \p NodePtrTy ordered when they're put into \p std::multiset.
  struct NodeCmpTy {
    bool operator()(const NodeTy &LHS, const NodeTy &RHS) const {
      return LHS.Size < RHS.Size;
    }
  };

  /// A \p FreeList is a set of Nodes. We're using \p std::multiset here to make
  /// the look up procedure more efficient.
  using FreeListTy = std::multiset<std::reference_wrapper<NodeTy>, NodeCmpTy>;

  /// A list of \p FreeListTy entries, each of which is a \p std::multiset of
  /// Nodes whose size is less or equal to a specific bucket size.
  std::vector<FreeListTy> FreeLists;
  /// A list of mutex for each \p FreeListTy entry
  std::vector<std::mutex> FreeListLocks;
  /// A table to map from a target pointer to its node
  std::unordered_map<void *, NodeTy> PtrToNodeTable;
  /// The mutex for the table \p PtrToNodeTable
  std::mutex MapTableLock;
  /// A reference to its corresponding \p DeviceTy object
  DeviceTy &Device;

  /// Request memory from target device
  void *allocateOnDevice(size_t Size, void *HstPtr) const;

  /// Deallocate data on device
  int deleteOnDevice(void *Ptr) const;

  /// This function is called when it tries to allocate memory on device but the
  /// device returns out of memory. It will first free all memory in the
  /// FreeList and try to allocate again.
  void *freeAndAllocate(size_t Size, void *HstPtr);

  /// The goal is to allocate memory on the device. It first tries to allocate
  /// directly on the device. If a \p nullptr is returned, it might be because
  /// the device is OOM. In that case, it will free all unused memory and then
  /// try again.
  void *allocateOrFreeAndAllocateOnDevice(size_t Size, void *HstPtr);

public:
  /// Constructor. If \p Threshold is non-zero, then the default threshold will
  /// be overwritten by \p Threshold.
  MemoryManagerTy(DeviceTy &Dev, size_t Threshold = 0);

  /// Destructor
  ~MemoryManagerTy();

  /// Allocate memory of size \p Size from target device. \p HstPtr is used to
  /// assist the allocation.
  void *allocate(size_t Size, void *HstPtr);

  /// Deallocate memory pointed by \p TgtPtr
  int free(void *TgtPtr);
};

#endif // LLVM_OPENMP_LIBOMPTARGET_SRC_MEMORYMANAGER_H
