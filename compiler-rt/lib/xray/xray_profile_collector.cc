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
#include "xray_allocator.h"
#include "xray_defs.h"
#include "xray_profiling_flags.h"
#include "xray_segmented_array.h"
#include <memory>
#include <pthread.h>
#include <utility>

namespace __xray {
namespace profileCollectorService {

namespace {

SpinMutex GlobalMutex;
struct ThreadTrie {
  tid_t TId;
  typename std::aligned_storage<sizeof(FunctionCallTrie)>::type TrieStorage;
};

struct ProfileBuffer {
  void *Data;
  size_t Size;
};

// Current version of the profile format.
constexpr u64 XRayProfilingVersion = 0x20180424;

// Identifier for XRay profiling files 'xrayprof' in hex.
constexpr u64 XRayMagicBytes = 0x7872617970726f66;

struct XRayProfilingFileHeader {
  const u64 MagicBytes = XRayMagicBytes;
  const u64 Version = XRayProfilingVersion;
  u64 Timestamp = 0; // System time in nanoseconds.
  u64 PID = 0;       // Process ID.
};

struct BlockHeader {
  u32 BlockSize;
  u32 BlockNum;
  u64 ThreadId;
};

using ThreadTriesArray = Array<ThreadTrie>;
using ProfileBufferArray = Array<ProfileBuffer>;
using ThreadTriesArrayAllocator = typename ThreadTriesArray::AllocatorType;
using ProfileBufferArrayAllocator = typename ProfileBufferArray::AllocatorType;

// These need to be global aligned storage to avoid dynamic initialization. We
// need these to be aligned to allow us to placement new objects into the
// storage, and have pointers to those objects be appropriately aligned.
static typename std::aligned_storage<sizeof(FunctionCallTrie::Allocators)>::type
    AllocatorStorage;
static typename std::aligned_storage<sizeof(ThreadTriesArray)>::type
    ThreadTriesStorage;
static typename std::aligned_storage<sizeof(ProfileBufferArray)>::type
    ProfileBuffersStorage;
static typename std::aligned_storage<sizeof(ThreadTriesArrayAllocator)>::type
    ThreadTriesArrayAllocatorStorage;
static typename std::aligned_storage<sizeof(ProfileBufferArrayAllocator)>::type
    ProfileBufferArrayAllocatorStorage;

static ThreadTriesArray *ThreadTries = nullptr;
static ThreadTriesArrayAllocator *ThreadTriesAllocator = nullptr;
static ProfileBufferArray *ProfileBuffers = nullptr;
static ProfileBufferArrayAllocator *ProfileBuffersAllocator = nullptr;
static FunctionCallTrie::Allocators *GlobalAllocators = nullptr;

} // namespace

void post(const FunctionCallTrie &T, tid_t TId) XRAY_NEVER_INSTRUMENT {
  static pthread_once_t Once = PTHREAD_ONCE_INIT;
  pthread_once(&Once, +[] { reset(); });

  ThreadTrie *Item = nullptr;
  {
    SpinMutexLock Lock(&GlobalMutex);
    if (GlobalAllocators == nullptr || ThreadTries == nullptr)
      return;

    Item = ThreadTries->Append({});
    Item->TId = TId;
    auto Trie = reinterpret_cast<FunctionCallTrie *>(&Item->TrieStorage);
    new (Trie) FunctionCallTrie(*GlobalAllocators);
  }

  auto Trie = reinterpret_cast<FunctionCallTrie *>(&Item->TrieStorage);
  T.deepCopyInto(*Trie);
}

// A PathArray represents the function id's representing a stack trace. In this
// context a path is almost always represented from the leaf function in a call
// stack to a root of the call trie.
using PathArray = Array<int32_t>;

struct ProfileRecord {
  using PathAllocator = typename PathArray::AllocatorType;

  // The Path in this record is the function id's from the leaf to the root of
  // the function call stack as represented from a FunctionCallTrie.
  PathArray Path;
  const FunctionCallTrie::Node *Node = nullptr;

  // Constructor for in-place construction.
  ProfileRecord(PathAllocator &A,
                const FunctionCallTrie::Node *N) XRAY_NEVER_INSTRUMENT
      : Path(A),
        Node(N) {}
};

namespace {

using ProfileRecordArray = Array<ProfileRecord>;

// Walk a depth-first traversal of each root of the FunctionCallTrie to generate
// the path(s) and the data associated with the path.
static void
populateRecords(ProfileRecordArray &PRs, ProfileRecord::PathAllocator &PA,
                const FunctionCallTrie &Trie) XRAY_NEVER_INSTRUMENT {
  using StackArray = Array<const FunctionCallTrie::Node *>;
  using StackAllocator = typename StackArray::AllocatorType;
  StackAllocator StackAlloc(profilingFlags()->stack_allocator_max);
  StackArray DFSStack(StackAlloc);
  for (const auto R : Trie.getRoots()) {
    DFSStack.Append(R);
    while (!DFSStack.empty()) {
      auto Node = DFSStack.back();
      DFSStack.trim(1);
      auto Record = PRs.AppendEmplace(PA, Node);
      if (Record == nullptr)
        return;
      DCHECK_NE(Record, nullptr);

      // Traverse the Node's parents and as we're doing so, get the FIds in
      // the order they appear.
      for (auto N = Node; N != nullptr; N = N->Parent)
        Record->Path.Append(N->FId);
      DCHECK(!Record->Path.empty());

      for (const auto C : Node->Callees)
        DFSStack.Append(C.NodePtr);
    }
  }
}

static void serializeRecords(ProfileBuffer *Buffer, const BlockHeader &Header,
                             const ProfileRecordArray &ProfileRecords)
    XRAY_NEVER_INSTRUMENT {
  auto NextPtr = static_cast<uint8_t *>(
                     internal_memcpy(Buffer->Data, &Header, sizeof(Header))) +
                 sizeof(Header);
  for (const auto &Record : ProfileRecords) {
    // List of IDs follow:
    for (const auto FId : Record.Path)
      NextPtr =
          static_cast<uint8_t *>(internal_memcpy(NextPtr, &FId, sizeof(FId))) +
          sizeof(FId);

    // Add the sentinel here.
    constexpr int32_t SentinelFId = 0;
    NextPtr = static_cast<uint8_t *>(
                  internal_memset(NextPtr, SentinelFId, sizeof(SentinelFId))) +
              sizeof(SentinelFId);

    // Add the node data here.
    NextPtr =
        static_cast<uint8_t *>(internal_memcpy(
            NextPtr, &Record.Node->CallCount, sizeof(Record.Node->CallCount))) +
        sizeof(Record.Node->CallCount);
    NextPtr = static_cast<uint8_t *>(
                  internal_memcpy(NextPtr, &Record.Node->CumulativeLocalTime,
                                  sizeof(Record.Node->CumulativeLocalTime))) +
              sizeof(Record.Node->CumulativeLocalTime);
  }

  DCHECK_EQ(NextPtr - static_cast<uint8_t *>(Buffer->Data), Buffer->Size);
}

} // namespace

void serialize() XRAY_NEVER_INSTRUMENT {
  SpinMutexLock Lock(&GlobalMutex);

  if (GlobalAllocators == nullptr || ThreadTries == nullptr ||
      ProfileBuffers == nullptr)
    return;

  // Clear out the global ProfileBuffers, if it's not empty.
  for (auto &B : *ProfileBuffers)
    deallocateBuffer(reinterpret_cast<uint8_t *>(B.Data), B.Size);
  ProfileBuffers->trim(ProfileBuffers->size());

  if (ThreadTries->empty())
    return;

  // Then repopulate the global ProfileBuffers.
  u32 I = 0;
  for (const auto &ThreadTrie : *ThreadTries) {
    using ProfileRecordAllocator = typename ProfileRecordArray::AllocatorType;
    ProfileRecordAllocator PRAlloc(profilingFlags()->global_allocator_max);
    ProfileRecord::PathAllocator PathAlloc(
        profilingFlags()->global_allocator_max);
    ProfileRecordArray ProfileRecords(PRAlloc);

    // First, we want to compute the amount of space we're going to need. We'll
    // use a local allocator and an __xray::Array<...> to store the intermediary
    // data, then compute the size as we're going along. Then we'll allocate the
    // contiguous space to contain the thread buffer data.
    const auto &Trie =
        *reinterpret_cast<const FunctionCallTrie *>(&(ThreadTrie.TrieStorage));
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
      CumulativeSizes += 20 + (4 * Record.Path.size());

    BlockHeader Header{16 + CumulativeSizes, I++, ThreadTrie.TId};
    auto Buffer = ProfileBuffers->Append({});
    Buffer->Size = sizeof(Header) + CumulativeSizes;
    Buffer->Data = allocateBuffer(Buffer->Size);
    DCHECK_NE(Buffer->Data, nullptr);
    serializeRecords(Buffer, Header, ProfileRecords);
  }
}

void reset() XRAY_NEVER_INSTRUMENT {
  SpinMutexLock Lock(&GlobalMutex);

  if (ProfileBuffers != nullptr) {
    // Clear out the profile buffers that have been serialized.
    for (auto &B : *ProfileBuffers)
      deallocateBuffer(reinterpret_cast<uint8_t *>(B.Data), B.Size);
    ProfileBuffers->trim(ProfileBuffers->size());
  }

  if (ThreadTries != nullptr) {
    // Clear out the function call tries per thread.
    for (auto &T : *ThreadTries) {
      auto Trie = reinterpret_cast<FunctionCallTrie *>(&T.TrieStorage);
      Trie->~FunctionCallTrie();
    }
    ThreadTries->trim(ThreadTries->size());
  }

  // Reset the global allocators.
  if (GlobalAllocators != nullptr)
    GlobalAllocators->~Allocators();

  GlobalAllocators =
      reinterpret_cast<FunctionCallTrie::Allocators *>(&AllocatorStorage);
  new (GlobalAllocators) FunctionCallTrie::Allocators();
  *GlobalAllocators = FunctionCallTrie::InitAllocators();

  if (ThreadTriesAllocator != nullptr)
    ThreadTriesAllocator->~ThreadTriesArrayAllocator();

  ThreadTriesAllocator = reinterpret_cast<ThreadTriesArrayAllocator *>(
      &ThreadTriesArrayAllocatorStorage);
  new (ThreadTriesAllocator)
      ThreadTriesArrayAllocator(profilingFlags()->global_allocator_max);
  ThreadTries = reinterpret_cast<ThreadTriesArray *>(&ThreadTriesStorage);
  new (ThreadTries) ThreadTriesArray(*ThreadTriesAllocator);

  if (ProfileBuffersAllocator != nullptr)
    ProfileBuffersAllocator->~ProfileBufferArrayAllocator();

  ProfileBuffersAllocator = reinterpret_cast<ProfileBufferArrayAllocator *>(
      &ProfileBufferArrayAllocatorStorage);
  new (ProfileBuffersAllocator)
      ProfileBufferArrayAllocator(profilingFlags()->global_allocator_max);
  ProfileBuffers =
      reinterpret_cast<ProfileBufferArray *>(&ProfileBuffersStorage);
  new (ProfileBuffers) ProfileBufferArray(*ProfileBuffersAllocator);
}

XRayBuffer nextBuffer(XRayBuffer B) XRAY_NEVER_INSTRUMENT {
  SpinMutexLock Lock(&GlobalMutex);

  if (ProfileBuffers == nullptr || ProfileBuffers->size() == 0)
    return {nullptr, 0};

  static pthread_once_t Once = PTHREAD_ONCE_INIT;
  static typename std::aligned_storage<sizeof(XRayProfilingFileHeader)>::type
      FileHeaderStorage;
  pthread_once(&Once,
               +[] { new (&FileHeaderStorage) XRayProfilingFileHeader{}; });

  if (UNLIKELY(B.Data == nullptr)) {
    // The first buffer should always contain the file header information.
    auto &FileHeader =
        *reinterpret_cast<XRayProfilingFileHeader *>(&FileHeaderStorage);
    FileHeader.Timestamp = NanoTime();
    FileHeader.PID = internal_getpid();
    return {&FileHeaderStorage, sizeof(XRayProfilingFileHeader)};
  }

  if (UNLIKELY(B.Data == &FileHeaderStorage))
    return {(*ProfileBuffers)[0].Data, (*ProfileBuffers)[0].Size};

  BlockHeader Header;
  internal_memcpy(&Header, B.Data, sizeof(BlockHeader));
  auto NextBlock = Header.BlockNum + 1;
  if (NextBlock < ProfileBuffers->size())
    return {(*ProfileBuffers)[NextBlock].Data,
            (*ProfileBuffers)[NextBlock].Size};
  return {nullptr, 0};
}

} // namespace profileCollectorService
} // namespace __xray
