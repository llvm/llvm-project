//===-- xray_profile_collector.cc ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of XRay, a dynamic runtime instrumentation system.
//
// This implements the interface for the profileCollectorService.
//
//===----------------------------------------------------------------------===//
#include "xray_profile_collector.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_vector.h"
#include "xray_profiling_flags.h"
#include <memory>
#include <pthread.h>
#include <utility>

namespace __xray {
namespace profileCollectorService {

namespace {

SpinMutex GlobalMutex;
struct ThreadTrie {
  tid_t TId;
  FunctionCallTrie *Trie;
};

struct ProfileBuffer {
  void *Data;
  size_t Size;
};

struct BlockHeader {
  u32 BlockSize;
  u32 BlockNum;
  u64 ThreadId;
};

// These need to be pointers that point to heap/internal-allocator-allocated
// objects because these are accessed even at program exit.
Vector<ThreadTrie> *ThreadTries = nullptr;
Vector<ProfileBuffer> *ProfileBuffers = nullptr;
FunctionCallTrie::Allocators *GlobalAllocators = nullptr;

} // namespace

void post(const FunctionCallTrie &T, tid_t TId) {
  static pthread_once_t Once = PTHREAD_ONCE_INIT;
  pthread_once(&Once, +[] {
    SpinMutexLock Lock(&GlobalMutex);
    GlobalAllocators = reinterpret_cast<FunctionCallTrie::Allocators *>(
        InternalAlloc(sizeof(FunctionCallTrie::Allocators)));
    new (GlobalAllocators) FunctionCallTrie::Allocators();
    *GlobalAllocators = FunctionCallTrie::InitAllocatorsCustom(
        profilingFlags()->global_allocator_max);
    ThreadTries = reinterpret_cast<Vector<ThreadTrie> *>(
        InternalAlloc(sizeof(Vector<ThreadTrie>)));
    new (ThreadTries) Vector<ThreadTrie>();
    ProfileBuffers = reinterpret_cast<Vector<ProfileBuffer> *>(
        InternalAlloc(sizeof(Vector<ProfileBuffer>)));
    new (ProfileBuffers) Vector<ProfileBuffer>();
  });
  DCHECK_NE(GlobalAllocators, nullptr);
  DCHECK_NE(ThreadTries, nullptr);
  DCHECK_NE(ProfileBuffers, nullptr);

  ThreadTrie *Item = nullptr;
  {
    SpinMutexLock Lock(&GlobalMutex);
    if (GlobalAllocators == nullptr)
      return;

    Item = ThreadTries->PushBack();
    Item->TId = TId;

    // Here we're using the internal allocator instead of the managed allocator
    // because:
    //
    // 1) We're not using the segmented array data structure to host
    //    FunctionCallTrie objects. We're using a Vector (from sanitizer_common)
    //    which works like a std::vector<...> keeping elements contiguous in
    //    memory. The segmented array data structure assumes that elements are
    //    trivially destructible, where FunctionCallTrie isn't.
    //
    // 2) Using a managed allocator means we need to manage that separately,
    //    which complicates the nature of this code. To get around that, we're
    //    using the internal allocator instead, which has its own global state
    //    and is decoupled from the lifetime management required by the managed
    //    allocator we have in XRay.
    //
    Item->Trie = reinterpret_cast<FunctionCallTrie *>(InternalAlloc(
        sizeof(FunctionCallTrie), nullptr, alignof(FunctionCallTrie)));
    DCHECK_NE(Item->Trie, nullptr);
    new (Item->Trie) FunctionCallTrie(*GlobalAllocators);
  }

  T.deepCopyInto(*Item->Trie);
}

// A PathArray represents the function id's representing a stack trace. In this
// context a path is almost always represented from the leaf function in a call
// stack to a root of the call trie.
using PathArray = Array<int32_t>;

struct ProfileRecord {
  using PathAllocator = typename PathArray::AllocatorType;

  // The Path in this record is the function id's from the leaf to the root of
  // the function call stack as represented from a FunctionCallTrie.
  PathArray *Path = nullptr;
  const FunctionCallTrie::Node *Node = nullptr;

  // Constructor for in-place construction.
  ProfileRecord(PathAllocator &A, const FunctionCallTrie::Node *N)
      : Path([&] {
          auto P =
              reinterpret_cast<PathArray *>(InternalAlloc(sizeof(PathArray)));
          new (P) PathArray(A);
          return P;
        }()),
        Node(N) {}
};

namespace {

using ProfileRecordArray = Array<ProfileRecord>;

// Walk a depth-first traversal of each root of the FunctionCallTrie to generate
// the path(s) and the data associated with the path.
static void populateRecords(ProfileRecordArray &PRs,
                            ProfileRecord::PathAllocator &PA,
                            const FunctionCallTrie &Trie) {
  using StackArray = Array<const FunctionCallTrie::Node *>;
  using StackAllocator = typename StackArray::AllocatorType;
  StackAllocator StackAlloc(profilingFlags()->stack_allocator_max, 0);
  StackArray DFSStack(StackAlloc);
  for (const auto R : Trie.getRoots()) {
    DFSStack.Append(R);
    while (!DFSStack.empty()) {
      auto Node = DFSStack.back();
      DFSStack.trim(1);
      auto Record = PRs.AppendEmplace(PA, Node);
      DCHECK_NE(Record, nullptr);

      // Traverse the Node's parents and as we're doing so, get the FIds in
      // the order they appear.
      for (auto N = Node; N != nullptr; N = N->Parent)
        Record->Path->Append(N->FId);
      DCHECK(!Record->Path->empty());

      for (const auto C : Node->Callees)
        DFSStack.Append(C.NodePtr);
    }
  }
}

static void serializeRecords(ProfileBuffer *Buffer, const BlockHeader &Header,
                             const ProfileRecordArray &ProfileRecords) {
  auto NextPtr = static_cast<char *>(
                     internal_memcpy(Buffer->Data, &Header, sizeof(Header))) +
                 sizeof(Header);
  for (const auto &Record : ProfileRecords) {
    // List of IDs follow:
    for (const auto FId : *Record.Path)
      NextPtr =
          static_cast<char *>(internal_memcpy(NextPtr, &FId, sizeof(FId))) +
          sizeof(FId);

    // Add the sentinel here.
    constexpr int32_t SentinelFId = 0;
    NextPtr = static_cast<char *>(
                  internal_memset(NextPtr, SentinelFId, sizeof(SentinelFId))) +
              sizeof(SentinelFId);

    // Add the node data here.
    NextPtr =
        static_cast<char *>(internal_memcpy(NextPtr, &Record.Node->CallCount,
                                            sizeof(Record.Node->CallCount))) +
        sizeof(Record.Node->CallCount);
    NextPtr = static_cast<char *>(
                  internal_memcpy(NextPtr, &Record.Node->CumulativeLocalTime,
                                  sizeof(Record.Node->CumulativeLocalTime))) +
              sizeof(Record.Node->CumulativeLocalTime);
  }

  DCHECK_EQ(NextPtr - static_cast<char *>(Buffer->Data), Buffer->Size);
}

} // namespace

void serialize() {
  SpinMutexLock Lock(&GlobalMutex);

  // Clear out the global ProfileBuffers.
  for (uptr I = 0; I < ProfileBuffers->Size(); ++I)
    InternalFree((*ProfileBuffers)[I].Data);
  ProfileBuffers->Reset();

  if (ThreadTries->Size() == 0)
    return;

  // Then repopulate the global ProfileBuffers.
  for (u32 I = 0; I < ThreadTries->Size(); ++I) {
    using ProfileRecordAllocator = typename ProfileRecordArray::AllocatorType;
    ProfileRecordAllocator PRAlloc(profilingFlags()->global_allocator_max, 0);
    ProfileRecord::PathAllocator PathAlloc(
        profilingFlags()->global_allocator_max, 0);
    ProfileRecordArray ProfileRecords(PRAlloc);

    // First, we want to compute the amount of space we're going to need. We'll
    // use a local allocator and an __xray::Array<...> to store the intermediary
    // data, then compute the size as we're going along. Then we'll allocate the
    // contiguous space to contain the thread buffer data.
    const auto &Trie = *(*ThreadTries)[I].Trie;
    if (Trie.getRoots().empty())
      continue;
    populateRecords(ProfileRecords, PathAlloc, Trie);
    DCHECK(!Trie.getRoots().empty());
    DCHECK(!ProfileRecords.empty());

    // Go through each record, to compute the sizes.
    //
    // header size = block size (4 bytes)
    //   + block number (4 bytes)
    //   + thread id (8 bytes)
    // record size = path ids (4 bytes * number of ids + sentinel 4 bytes)
    //   + call count (8 bytes)
    //   + local time (8 bytes)
    //   + end of record (8 bytes)
    u32 CumulativeSizes = 0;
    for (const auto &Record : ProfileRecords)
      CumulativeSizes += 20 + (4 * Record.Path->size());

    BlockHeader Header{16 + CumulativeSizes, I, (*ThreadTries)[I].TId};
    auto Buffer = ProfileBuffers->PushBack();
    Buffer->Size = sizeof(Header) + CumulativeSizes;
    Buffer->Data = InternalAlloc(Buffer->Size, nullptr, 64);
    DCHECK_NE(Buffer->Data, nullptr);
    serializeRecords(Buffer, Header, ProfileRecords);

    // Now clean up the ProfileRecords array, one at a time.
    for (auto &Record : ProfileRecords) {
      Record.Path->~PathArray();
      InternalFree(Record.Path);
    }
  }
}

void reset() {
  SpinMutexLock Lock(&GlobalMutex);
  if (ProfileBuffers != nullptr) {
    // Clear out the profile buffers that have been serialized.
    for (uptr I = 0; I < ProfileBuffers->Size(); ++I)
      InternalFree((*ProfileBuffers)[I].Data);
    ProfileBuffers->Reset();
    InternalFree(ProfileBuffers);
    ProfileBuffers = nullptr;
  }

  if (ThreadTries != nullptr) {
    // Clear out the function call tries per thread.
    for (uptr I = 0; I < ThreadTries->Size(); ++I) {
      auto &T = (*ThreadTries)[I];
      T.Trie->~FunctionCallTrie();
      InternalFree(T.Trie);
    }
    ThreadTries->Reset();
    InternalFree(ThreadTries);
    ThreadTries = nullptr;
  }

  // Reset the global allocators.
  if (GlobalAllocators != nullptr) {
    GlobalAllocators->~Allocators();
    InternalFree(GlobalAllocators);
    GlobalAllocators = nullptr;
  }
  GlobalAllocators = reinterpret_cast<FunctionCallTrie::Allocators *>(
      InternalAlloc(sizeof(FunctionCallTrie::Allocators)));
  new (GlobalAllocators) FunctionCallTrie::Allocators();
  *GlobalAllocators = FunctionCallTrie::InitAllocators();
  ThreadTries = reinterpret_cast<Vector<ThreadTrie> *>(
      InternalAlloc(sizeof(Vector<ThreadTrie>)));
  new (ThreadTries) Vector<ThreadTrie>();
  ProfileBuffers = reinterpret_cast<Vector<ProfileBuffer> *>(
      InternalAlloc(sizeof(Vector<ProfileBuffer>)));
  new (ProfileBuffers) Vector<ProfileBuffer>();
}

XRayBuffer nextBuffer(XRayBuffer B) {
  SpinMutexLock Lock(&GlobalMutex);

  if (ProfileBuffers == nullptr || ProfileBuffers->Size() == 0)
    return {nullptr, 0};

  if (B.Data == nullptr)
    return {(*ProfileBuffers)[0].Data, (*ProfileBuffers)[0].Size};

  BlockHeader Header;
  internal_memcpy(&Header, B.Data, sizeof(BlockHeader));
  auto NextBlock = Header.BlockNum + 1;
  if (NextBlock < ProfileBuffers->Size())
    return {(*ProfileBuffers)[NextBlock].Data,
            (*ProfileBuffers)[NextBlock].Size};
  return {nullptr, 0};
}

} // namespace profileCollectorService
} // namespace __xray
