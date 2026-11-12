//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// One region per size class. Class n holds chunks of 2^(6+n) bytes.
/// Chunks grow up from the base; one 8-byte entry per chunk grows down
/// from the top. Distinct from the existing size-class allocator.
///
///   ┌───────────────────────────────────────────────┬──────────────────┐
///   │ c0    c1    c2    ...                         │     ... e2 e1 e0 │
///   └───────────────────────────────────────────────┴──────────────────┘
///                                          chunks →   ← entries
///
/// An allocation is padded with redzones to prevent overruns into adjacent
/// objects.
///
///         left                                               right
///   ┌───────────────┐   ┌──────────────────────────┐   ┌───────────────┐
///   │    redzone    │...│          memory          │...│    redzone    │
///   └───────────────┘   └──────────────────────────┘   └───────────────┘
///
/// The allocation is tracked by a metadata entry calculated from the allocation
/// address itself. The signed offset is used to skip the leading redzone and
/// can also be made negative to span multiple chunks.
///
///   ┌──────────────────────────┬─┬─────────────────────────────────────┐
///   │ signed offset         26 │p│ size                             37 │
///   └──────────────────────────┴─┴─────────────────────────────────────┘
///    63                     38 37 36                                  0
///
/// Freed ranges are quarantined and poisoned to enable detection of
/// use-after-free errors.
///
//===----------------------------------------------------------------------===//

#ifndef DASAN_ALLOCATOR_H
#define DASAN_ALLOCATOR_H

#include "dasan_mapping.h"
#include "dasan_platform.h"
#include "sanitizer_common/sanitizer_common.h"

namespace __dasan {

inline constexpr uptr kMaxRedzone = 2048;

// The amount of leading redzone to use for a given allocation size.
inline uptr LeadingRedzone(uptr Size, uptr ChunkSize, uptr Alignment) {
  if (!Alignment)
    Alignment = kAllocAlignment;
  if (Size + kMinRedzone >= ChunkSize)
    return 0;
  uptr Off = RoundDownTo(ChunkSize - Size - kMinRedzone, Alignment);
  return Off > kMaxRedzone ? RoundDownTo(kMaxRedzone, Alignment) : Off;
}

// Allocations from the HSA runtime have a minimum granule. Ensure we are
// aligned on that boundary.
inline uptr HeapAlign(uptr Size) {
  const uptr Granule = VaGranule();
  return Size >= Granule ? Granule : kAllocAlignment;
}

// Smallest class whose one chunk holds Size at Alignment.
inline uptr ChooseClass(uptr Size, uptr Alignment) {
  if (!Size || Size > kSizeMask)
    return kNumSizeClasses;
  if (!Alignment)
    Alignment = kAllocAlignment;
  for (uptr ClassId = 0; ClassId < kNumSizeClasses; ++ClassId) {
    const uptr Chunk = ClassIdToSize(ClassId);
    // The chunk base is aligned to the chunk. A smaller chunk cannot hold
    // an Alignment-aligned pointer.
    if (Chunk < Alignment)
      continue;
    const uptr Off = LeadingRedzone(Size, Chunk, Alignment);
    if (Off + Size + kMinRedzone <= Chunk)
      return ClassId;
  }
  return kNumSizeClasses;
}

// Tracking state for a single allocation.
struct AllocInfo {
  uptr Beg;
  uptr Size;
  hsa_amd_memory_pool_t Pool;
  bool Freed;
  u32 AllocStack;
  u32 FreeStack;
  u32 AllocTid;
  u32 FreeTid;
};

// Segregated fit size class allocator.
class Allocator {
 public:
  bool Init();
  void Shutdown();

  bool IsPlaced(uptr Addr) const { return Reserved && PointerIsMine(Addr); }

  // `Size` bytes in class `ClassId`. MapUser is false for a run the caller
  // maps. The pointer to free, or 0 when this class cannot hold it.
  uptr Allocate(hsa_amd_memory_pool_t Pool, uptr ClassId, uptr Size,
                u32 AllocFlags = 0, u32 Stack = 0, u32 Tid = 0,
                bool MapUser = true);

  // Free a pointer Allocate returned for a mapped object.
  bool Deallocate(uptr Addr, u32 Stack = 0, u32 Tid = 0);

  // Free the base Allocate returned for an unmapped run.
  void Release(uptr Addr);

  // Write the metadata associated with an existing allocation.
  bool Publish(uptr Addr, uptr Size, bool Pad = false);

  // Allow agents to access the allocation and metadata given the address.
  bool AllowAccess(uptr Addr, const hsa_agent_t* Agents, u32 N);

  // Return the information associated with this address for reporting.
  bool getAllocInfo(uptr Addr, AllocInfo* Out) const;

  struct Counters {
    uptr Placed;
    uptr LargerClass;
    uptr Freed;
    uptr TooLarge;
    uptr Exhausted;
    uptr CodeObjects;
    uptr CodeObjectChunks;
  } Count;

 private:
  struct ChunkState {
    u64 UserSize;
    u64 Mapped;
    uptr UserOffset;
    uptr RunStart;
    uptr RunLen;
    u32 List;
    u32 AllocStack;
    u32 FreeStack;
    u32 AllocTid;
    u32 FreeTid;
    bool Live;
    // Skipped so a device's metadata, or a user mapping, starts on a granule
    // the previous one does not own.
    bool Gap;
    // Mapped, redzone, quarantined. An unmapped run has none of these.
    bool Heap;
  };

  // Heap object. Offset is from the base of its chunk.
  struct Placement {
    uptr Offset;
    uptr UserSize;
    u32 Stack;
    u32 Tid;
  };

  // A contiguous run of free chunks. SizeClassAllocator64 keeps one flat
  // array of independent chunks in the VA. Chunks here are backed by one
  // pool and an allocation is a run, so the list is per device, in
  // ordinary memory, and each element is a run.
  struct Run {
    uptr Begin;
    uptr Count;
    bool UserMapped;
  };

  struct FreeList {
    hsa_amd_memory_pool_t Pool;
    InternalMmapVectorNoCtor<Run> Free;
    InternalMmapVectorNoCtor<Run> Quarantine;
    uptr Head;
    uptr QuarantinedBytes;
  };

  // One record for a run the caller maps. Begin is its first chunk; Count
  // is how many chunks the mapping occupies.
  struct Reservation {
    uptr Begin;
    uptr Count;
    uptr List;
    uptr UserSize;
  };

  struct Region {
    InternalMmapVectorNoCtor<ChunkState> Chunks;
    InternalMmapVectorNoCtor<Reservation> Reservations;
    InternalMmapVectorNoCtor<FreeList> FreeLists;
    uptr NextChunk;
    uptr MetadataBytes;
    hsa_amd_memory_pool_t MetadataPool;
    bool MetadataPoolSet;
  };

  struct Mapping {
    uptr Addr;
    uptr Size;
  };

  Region Regions[kNumSizeClasses];
  InternalMmapVectorNoCtor<u64> VmemHandles;
  InternalMmapVectorNoCtor<Mapping> Maps;
  uptr Granule;
  uptr MetaGranule;
  uptr QuarantinedBytes;

  bool Initialized;
  bool Reserved;
  bool MetadataOk;

  uptr ChunkBegin(uptr ClassId, uptr ChunkIdx) const {
    return RegionBeg(ClassId) + ChunkIdx * ClassIdToSize(ClassId);
  }
  uptr MetadataBytesFor(uptr Chunks) const {
    return RoundUpTo(Chunks * kMetadataSize, MetaGranule);
  }
  uptr MetadataBegin(uptr ClassId, uptr Bytes) const {
    return RegionBeg(ClassId) + kRegionSize - Bytes;
  }
  // kUserMapSize in SizeClassAllocator64. The floor is the device granule:
  // 2 MB on current parts, so a small class is one handle per run.
  bool MapOnPopulate(uptr ClassId) const {
    return ClassIdToSize(ClassId) <= Granule;
  }
  uptr UserMapChunks(uptr ClassId) const;

  uptr AlignOf(uptr ClassId, uptr Alignment) const;

  void SkipToFreshGranule(uptr ClassId, hsa_amd_memory_pool_t Pool);
  void MarkGap(uptr ClassId, uptr Begin, uptr End);

  uptr FreeListFor(uptr ClassId, hsa_amd_memory_pool_t Pool);
  void PushFree(uptr ClassId, uptr List, Run R);
  bool TakeFree(uptr ClassId, uptr List, uptr Need, uptr Stride, bool MapUser,
                uptr* Begin);

  bool CommitMap(uptr Addr, uptr Bytes, u64 Handle, hsa_amd_memory_pool_t Home);
  void RollbackMap();

  bool CreateMetadata(hsa_amd_memory_pool_t Prefer, uptr Bytes, u64* Handle,
                      hsa_amd_memory_pool_t* Home);
  bool EnsureMetadata(hsa_amd_memory_pool_t Pool, uptr ClassId, uptr Chunks);
  bool MapUserChunk(hsa_amd_memory_pool_t Pool, uptr ClassId, uptr ChunkIdx,
                    uptr Need, u32 AllocFlags);
  bool MapPrefix(hsa_amd_memory_pool_t Pool, uptr ClassId, uptr Begin,
                 uptr Count, uptr Bytes, u32 AllocFlags);

  bool Grow(hsa_amd_memory_pool_t Pool, uptr ClassId, uptr List, uptr Need,
            uptr Stride, bool MapUser, u32 AllocFlags, uptr* Begin);

  bool WriteMetadata(uptr Addr, uptr Size, bool Pad, bool Poison);
  void MarkChunk(uptr ClassId, uptr Begin, uptr List, const Placement& Place);
  void ClearChunk(uptr ClassId, uptr Begin);

  bool Recycle(uptr ClassId, uptr List);
  bool RecycleOne(uptr PreferClass, uptr PreferList);
  void Quarantine(uptr ClassId, uptr List, Run R, u32 Stack, u32 Tid);
};

Allocator& GetAllocator();

}  // namespace __dasan

#endif  // DASAN_ALLOCATOR_H
