//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Device address sanitizer allocator.
///
//===----------------------------------------------------------------------===//

#include "dasan_allocator.h"

#include "dasan_flags.h"
#include "dasan_report.h"
#include "sanitizer_common/sanitizer_common.h"

using namespace __sanitizer;

namespace __dasan {

static const uptr kUserMapSize = 1 << 21;

static_assert(kMaxRedzone <= kOffsetMax, "a redzone can outgrow its entry");

uptr Allocator::UserMapChunks(uptr ClassId) const {
  uptr Bytes = Granule > kUserMapSize ? Granule : kUserMapSize;
  return Bytes / ClassIdToSize(ClassId);
}

uptr Allocator::AlignOf(uptr ClassId, uptr Alignment) const {
  const uptr Chunk = ClassIdToSize(ClassId);
  if (!Alignment || Alignment < Chunk)
    Alignment = Chunk;
  if (!IsPowerOfTwo(Alignment))
    Alignment = RoundUpToPowerOfTwo(Alignment);
  if (Alignment < Chunk)
    Alignment = Chunk;
  return Alignment;
}

void Allocator::SkipToFreshGranule(uptr ClassId, hsa_amd_memory_pool_t Pool) {
  Region& R = Regions[ClassId];
  const uptr Stride = MetaGranule / kMetadataSize;

  if (R.NextChunk != 0 && R.MetadataPoolSet && R.MetadataPool != Pool) {
    uptr Boundary = RoundUpTo(R.NextChunk, Stride);
    // Boundary's granule is already homed on the other device.
    if (R.MetadataBytes >= MetadataBytesFor(Boundary + 1))
      Boundary += Stride;
    R.NextChunk = Boundary;
  }
  R.MetadataPool = Pool;
  R.MetadataPoolSet = true;
  // The skipped index is a hole: nothing hands it out.
  if (R.NextChunk <= R.Chunks.size())
    return;

  const uptr Mark = R.Chunks.size();
  R.Chunks.resize(R.NextChunk);
  for (uptr I = Mark; I < R.NextChunk; ++I) R.Chunks[I].Gap = true;
}

void Allocator::MarkGap(uptr ClassId, uptr Begin, uptr End) {
  if (Begin >= End)
    return;

  Region& R = Regions[ClassId];
  if (R.Chunks.size() < End)
    R.Chunks.resize(End);
  for (uptr I = Begin; I < End; ++I) R.Chunks[I].Gap = true;

  if (R.NextChunk < End)
    R.NextChunk = End;
}

uptr Allocator::FreeListFor(uptr ClassId, hsa_amd_memory_pool_t Pool) {
  Region& R = Regions[ClassId];
  for (uptr I = 0; I < R.FreeLists.size(); ++I)
    if (R.FreeLists[I].Pool == Pool)
      return I;
  uptr I = R.FreeLists.size();
  R.FreeLists.resize(I + 1);
  R.FreeLists[I].Pool = Pool;
  R.FreeLists[I].Head = 0;
  R.FreeLists[I].QuarantinedBytes = 0;
  return I;
}

void Allocator::PushFree(uptr ClassId, uptr List, Run R) {
  if (!R.Count)
    return;
  auto& Free = Regions[ClassId].FreeLists[List].Free;
  for (uptr I = 0; I < Free.size();) {
    Run& F = Free[I];
    const bool Abut =
        F.Begin + F.Count == R.Begin || R.Begin + R.Count == F.Begin;
    if (!Abut || F.UserMapped != R.UserMapped) {
      ++I;
      continue;
    }
    if (F.Begin < R.Begin)
      R.Begin = F.Begin;
    R.Count += F.Count;
    F = Free.back();
    Free.pop_back();
    I = 0;
  }
  Free.push_back(R);
}

bool Allocator::TakeFree(uptr ClassId, uptr List, uptr Need, uptr Stride,
                         bool MapUser, uptr* Begin) {
  auto& Free = Regions[ClassId].FreeLists[List].Free;
  for (uptr N = Free.size(); N-- > 0;) {
    const Run R = Free[N];
    if (!MapUser && R.UserMapped)
      continue;
    const uptr Aligned = RoundUpTo(R.Begin, Stride);
    if (Aligned < R.Begin || Aligned - R.Begin > R.Count)
      continue;
    if (Need > R.Count - (Aligned - R.Begin))
      continue;
    Free[N] = Free.back();
    Free.pop_back();
    if (Aligned > R.Begin)
      PushFree(ClassId, List, {R.Begin, Aligned - R.Begin, R.UserMapped});
    const uptr End = R.Begin + R.Count;
    if (Aligned + Need < End)
      PushFree(ClassId, List,
               {Aligned + Need, End - (Aligned + Need), R.UserMapped});
    *Begin = Aligned;
    return true;
  }
  return false;
}

bool Allocator::CommitMap(uptr Addr, uptr Bytes, u64 Handle,
                          hsa_amd_memory_pool_t Home) {
  if (!VaMap(Addr, Bytes, Handle, Home)) {
    VaDestroy(Handle);
    return false;
  }
  Maps.push_back({Addr, Bytes});
  VmemHandles.push_back(Handle);
  return true;
}

void Allocator::RollbackMap() {
  const Mapping M = Maps.back();
  Maps.pop_back();
  VaUnmap(M.Addr, M.Size);
  VaDestroy(VmemHandles.back());
  VmemHandles.pop_back();
}

bool Allocator::CreateMetadata(hsa_amd_memory_pool_t Prefer, uptr Bytes,
                               u64* Handle, hsa_amd_memory_pool_t* Home) {
  if (HasPool(Prefer) && VaCreate(Prefer, Bytes, 0, Handle)) {
    *Home = Prefer;
    return true;
  }
  InternalMmapVector<hsa_amd_memory_pool_t> Pools;
  PlaceablePools(&Pools);
  for (hsa_amd_memory_pool_t Other : Pools) {
    if (!HasPool(Other) || Other == Prefer)
      continue;
    if (VaCreate(Other, Bytes, 0, Handle)) {
      *Home = Other;
      return true;
    }
  }

  if (VaCreate({}, Bytes, 0, Handle)) {
    *Home = {};
    return true;
  }
  return false;
}

bool Allocator::EnsureMetadata(hsa_amd_memory_pool_t Pool, uptr ClassId,
                               uptr Chunks) {
  Region& R = Regions[ClassId];
  // Cover these chunks only. SkipToFreshGranule keeps another device's
  // indices out of a granule this one already maps.
  uptr Want = MetadataBytesFor(Chunks);
  const uptr Max = MetadataBytesFor(ReachableChunks(ClassId));
  if (Want > Max)
    Want = Max;
  if (R.MetadataBytes >= Want) {
    return !HasPool(Pool) || VaAllow(MetadataBegin(ClassId, R.MetadataBytes),
                                     R.MetadataBytes, Pool);
  }

  const uptr Beg = MetadataBegin(ClassId, Want);
  const uptr Bytes = Want - R.MetadataBytes;
  u64 H = 0;
  hsa_amd_memory_pool_t Home = Pool;
  if (!CreateMetadata(Pool, Bytes, &H, &Home))
    return false;
  if (!CommitMap(Beg, Bytes, H, Home))
    return false;
  if (HasPool(Pool) && Pool != Home && !VaAllow(Beg, Bytes, Pool)) {
    RollbackMap();
    return false;
  }

  if (!MetadataOk && !VaProbe(Beg)) {
    Report("ERROR: %s: could not initialize metadata\n", SanitizerToolName);
    Die();
  }

  VaPublish(Beg, nullptr, Bytes);
  MetadataOk = true;
  R.MetadataBytes = Want;
  VReport(2, "%s: class %zu holds %zu KiB of entries%s\n", SanitizerToolName,
          ClassId, Want >> 10,
          Home != Pool ? " (spilled to another pool)" : "");
  return true;
}

bool Allocator::MapUserChunk(hsa_amd_memory_pool_t Pool, uptr ClassId,
                             uptr ChunkIdx, uptr Need, u32 AllocFlags) {
  ChunkState& S = Regions[ClassId].Chunks[ChunkIdx];
  const uptr Chunk = ClassIdToSize(ClassId);
  uptr Want = RoundUpTo(Need, Granule);
  if (Want > Chunk)
    Want = Chunk;
  if (S.Mapped >= Want)
    return true;

  const uptr Beg = ChunkBegin(ClassId, ChunkIdx) + S.Mapped;
  const uptr Bytes = Want - S.Mapped;
  u64 H = 0;
  if (!VaCreate(Pool, Bytes, AllocFlags, &H))
    return false;
  if (!CommitMap(Beg, Bytes, H, Pool))
    return false;
  S.Mapped = Want;
  return true;
}

bool Allocator::MapPrefix(hsa_amd_memory_pool_t Pool, uptr ClassId, uptr Begin,
                          uptr Count, uptr Bytes, u32 AllocFlags) {
  const uptr Chunk = ClassIdToSize(ClassId);

  // A small class was mapped as a batch. A short prefix here would be a
  // granule the device will not take.
  if (MapOnPopulate(ClassId)) {
    for (uptr I = 0; I < Count && Bytes; ++I) {
      const uptr Need = Bytes > Chunk ? Chunk : Bytes;
      if (Regions[ClassId].Chunks[Begin + I].Mapped < Need)
        return false;
      Bytes -= Need;
    }
    return true;
  }
  for (uptr I = 0; I < Count && Bytes; ++I) {
    const uptr Need = Bytes > Chunk ? Chunk : Bytes;
    if (!MapUserChunk(Pool, ClassId, Begin + I, Need, AllocFlags))
      return false;
    Bytes -= Need;
  }
  return true;
}

bool Allocator::Grow(hsa_amd_memory_pool_t Pool, uptr ClassId, uptr List,
                     uptr Need, uptr Stride, bool MapUser, u32 AllocFlags,
                     uptr* Begin) {
  Region& R = Regions[ClassId];
  SkipToFreshGranule(ClassId, Pool);

  const uptr Max = MaxUsableChunks(ClassId);
  const uptr Chunk = ClassIdToSize(ClassId);
  const bool Batch = MapUser && MapOnPopulate(ClassId);
  if (Batch) {
    const uptr Per = Granule / Chunk;
    const uptr MapAt = RoundUpTo(R.NextChunk, Per);
    // The next granule would land in the metadata.
    if (MapAt > Max)
      return false;
    if (MapAt > R.NextChunk)
      MarkGap(ClassId, R.NextChunk, MapAt);
  }

  if (R.NextChunk > Max)
    return false;
  const uptr First = RoundUpTo(R.NextChunk, Stride);
  if (First < R.NextChunk || First >= Max || Need > Max - First)
    return false;

  uptr BatchCount = Need;
  uptr MapFrom = First;
  if (Batch) {
    const uptr Per = Granule / Chunk;
    const uptr Span = First + Need - R.NextChunk;
    BatchCount = UserMapChunks(ClassId);
    if (BatchCount < Span)
      BatchCount = RoundUpTo(Span, Per);
    if (BatchCount > Max - R.NextChunk)
      BatchCount = (Max - R.NextChunk) / Per * Per;
    if (!BatchCount || BatchCount < Span)
      return false;
    MapFrom = R.NextChunk;
  }

  const uptr End = MapFrom + BatchCount;
  if (!EnsureMetadata(Pool, ClassId, End))
    return false;

  // The caller maps this run. One reservation names it.
  if (!MapUser) {
    if (First > R.NextChunk)
      PushFree(ClassId, List, {R.NextChunk, First - R.NextChunk, false});
    R.NextChunk = End;
    *Begin = First;
    return true;
  }

  if (Batch) {
    u64 H = 0;
    if (!VaCreate(Pool, BatchCount * Chunk, AllocFlags, &H))
      return false;
    if (!CommitMap(ChunkBegin(ClassId, MapFrom), BatchCount * Chunk, H, Pool))
      return false;
  }

  if (R.Chunks.size() < End)
    R.Chunks.resize(End);
  for (uptr I = R.NextChunk; I < End; ++I) {
    ChunkState& S = R.Chunks[I];
    S.List = static_cast<u32>(List);
    // The batch map starts at NextChunk. A large class maps only the
    // prefix the caller asked for, after Grow returns.
    if (Batch && I >= MapFrom)
      S.Mapped = Chunk;
  }

  // Alignment padding. Mapped when it sits inside the batch, otherwise a
  // bare reservation a later run can claim.
  if (First > R.NextChunk)
    PushFree(ClassId, List, {R.NextChunk, First - R.NextChunk, Batch});
  if (First + Need < End)
    PushFree(ClassId, List, {First + Need, End - (First + Need), Batch});
  R.NextChunk = End;
  *Begin = First;
  return true;
}

// Same function as globalRedzoneSize in the DeviceAddressSanitizer pass.
static uptr GlobalRedzoneBytes(uptr Size) {
  const uptr Chunk = ClassIdToSize(0);
  const uptr Max = 1ull << 18;
  if (Size <= Chunk / 2)
    return Chunk - Size;
  uptr RZ = (Size / Chunk / 4) * Chunk;
  if (RZ < Chunk)
    RZ = Chunk;
  if (RZ > Max)
    RZ = Max;
  if (Size % Chunk)
    RZ += Chunk - (Size % Chunk);
  return RZ;
}

static uptr CoveredBytes(uptr Size) {
  return RoundUpTo(Size + GlobalRedzoneBytes(Size), ClassIdToSize(0));
}

bool Allocator::Publish(uptr Addr, uptr Size, bool Pad) {
  return WriteMetadata(Addr, Size, Pad, false);
}

bool Allocator::WriteMetadata(uptr Addr, uptr Size, bool Pad, bool Poison) {
  if (!Size || !MetadataOk || Size > kSizeMask || !PointerIsMine(Addr))
    return false;
  const uptr ClassId = GetSizeClass(Addr);
  const uptr Chunk = ClassIdToSize(ClassId);
  uptr Base = 0;
  uptr Off = 0;
  uptr Count = 0;

  // A heap chunk stores its redzone offset. Any other address is the base of
  // a range the caller published.
  const uptr Idx = GetChunkIdx(Addr, ClassId);
  Region& R = Regions[ClassId];
  if (Idx < R.Chunks.size() && !R.Chunks[Idx].Gap && R.Chunks[Idx].Heap &&
      R.Chunks[Idx].RunLen) {
    const ChunkState& S = R.Chunks[Idx];
    Base = ChunkBegin(ClassId, S.RunStart);
    Off = S.UserOffset;
    if (Addr != Base + Off)
      return false;
    Count = S.RunLen;
  } else {
    if (Addr % Chunk)
      return false;
    Base = Addr;
    const uptr Bytes = Pad ? CoveredBytes(Size) : RoundUpTo(Size, Chunk);
    Count = Bytes / Chunk;
  }
  if (!Count || Base % Chunk || !PointerIsMine(Base) ||
      GetSizeClass(Base) != ClassId)
    return false;
  const sptr Last =
      static_cast<sptr>(Off) - static_cast<sptr>((Count - 1) * Chunk);
  if (!OffsetFits(static_cast<sptr>(Off)) || !OffsetFits(Last))
    return false;

  auto Entry = [&](uptr I) {
    const sptr Delta = static_cast<sptr>(Off) - static_cast<sptr>(I * Chunk);
    u64 E = MakeEntry(Poison ? 0 : Size, Delta);
    return Poison ? PoisonEntry(E) : E;
  };
  const uptr First = GetChunkIdx(Base, ClassId);
  if (Count == 1) {
    u64 E = Entry(0);
    VaPublish(GetMetadata(RegionBeg(ClassId), First), &E, sizeof(E));
    return true;
  }
  // The tail grows downward, so the last chunk is the lowest address.
  InternalMmapVector<u64> Ordered(Count);
  for (uptr I = 0; I < Count; ++I) Ordered[Count - 1 - I] = Entry(I);
  VaPublish(GetMetadata(RegionBeg(ClassId), First + Count - 1), Ordered.data(),
            Count * sizeof(u64));
  return true;
}

void Allocator::MarkChunk(uptr ClassId, uptr Begin, uptr List,
                          const Placement& Place) {
  ChunkState& S = Regions[ClassId].Chunks[Begin];
  S.List = static_cast<u32>(List);
  S.Live = true;
  S.Gap = false;
  S.Heap = true;
  S.RunStart = Begin;
  S.RunLen = 1;
  S.UserOffset = Place.Offset;
  S.UserSize = Place.UserSize;
  S.AllocStack = Place.Stack;
  S.FreeStack = 0;
  S.AllocTid = Place.Tid;
  S.FreeTid = 0;
}

void Allocator::ClearChunk(uptr ClassId, uptr Begin) {
  ChunkState& S = Regions[ClassId].Chunks[Begin];
  S.Live = false;
  S.Heap = false;
  S.UserSize = 0;
  S.UserOffset = 0;
  S.RunStart = 0;
  S.RunLen = 0;
  S.AllocStack = S.FreeStack = 0;
  S.AllocTid = S.FreeTid = 0;
}

bool Allocator::Recycle(uptr ClassId, uptr List) {
  FreeList& F = Regions[ClassId].FreeLists[List];
  if (F.Head >= F.Quarantine.size())
    return false;
  const Run R = F.Quarantine[F.Head++];
  const uptr Bytes = R.Count * ClassIdToSize(ClassId);
  F.QuarantinedBytes -= Bytes;
  QuarantinedBytes -= Bytes;
  if (F.Head == F.Quarantine.size()) {
    F.Quarantine.clear();
    F.Head = 0;
  }
  // Leave the chunk's record in place until a new run overwrites it, so a
  // use-after-free still names the allocation that owned these bytes.
  PushFree(ClassId, List, R);
  return true;
}

bool Allocator::RecycleOne(uptr PreferClass, uptr PreferList) {
  if (Recycle(PreferClass, PreferList))
    return true;
  for (uptr C = 0; C < kNumSizeClasses; ++C) {
    Region& R = Regions[C];
    for (uptr L = 0; L < R.FreeLists.size(); ++L) {
      if (C == PreferClass && L == PreferList)
        continue;
      if (Recycle(C, L))
        return true;
    }
  }
  return false;
}

void Allocator::Quarantine(uptr ClassId, uptr List, Run R, u32 Stack, u32 Tid) {
  Region& Reg = Regions[ClassId];
  const uptr Chunk = ClassIdToSize(ClassId);
  ChunkState& S = Reg.Chunks[R.Begin];
  const uptr UserOff = S.UserOffset;
  const uptr UserSize = S.UserSize;
  S.Live = false;
  S.FreeStack = Stack;
  S.FreeTid = Tid;
  WriteMetadata(ChunkBegin(ClassId, R.Begin) + UserOff, UserSize, false, true);

  FreeList& F = Reg.FreeLists[List];
  const uptr Bytes = R.Count * Chunk;
  F.Quarantine.push_back(R);
  F.QuarantinedBytes += Bytes;
  QuarantinedBytes += Bytes;
  const uptr Budget = (uptr)flags()->quarantine_size_mb << 20;
  while (QuarantinedBytes > Budget && RecycleOne(ClassId, List)) {
  }
}

bool Allocator::Init() {
  if (Initialized)
    return Reserved;
  Initialized = true;

  Granule = VaGranule();
  MetaGranule = VaMetaGranule();
  // Both are non-zero powers of two. The device granule covers a host page,
  // so every smaller class divides it. The metadata granule tiles an entry
  // and the metadata region.
  const uptr Page = GetPageSizeCached();
  if (!Granule || !MetaGranule || !IsPowerOfTwo(Granule) ||
      !IsPowerOfTwo(MetaGranule) || Granule < Page ||
      MetaGranule % kMetadataSize || kMetadataBytes % MetaGranule) {
    VReport(1, "%s: no usable metadata granule; HSA calls pass through\n",
            SanitizerToolName);
    return false;
  }

  if (!VaReserve(kSpaceBeg, kSpaceUsed)) {
    VReport(1, "%s: could not reserve [0x%zx, 0x%zx); HSA calls pass through\n",
            SanitizerToolName, (uptr)kSpaceBeg, (uptr)(kSpaceBeg + kSpaceUsed));
    return false;
  }
  Reserved = true;
  VReport(1, "%s: reserved [0x%zx, 0x%zx)\n", SanitizerToolName,
          (uptr)kSpaceBeg, (uptr)(kSpaceBeg + kSpaceUsed));
  return true;
}

void Allocator::Shutdown() {
  if (Reserved) {
    for (uptr I = 0; I < Maps.size(); ++I) VaUnmap(Maps[I].Addr, Maps[I].Size);
    Maps.clear();
    for (uptr I = 0; I < VmemHandles.size(); ++I) VaDestroy(VmemHandles[I]);
    VmemHandles.clear();

    for (uptr C = 0; C < kNumSizeClasses; ++C) {
      Region& R = Regions[C];
      for (uptr I = 0; I < R.FreeLists.size(); ++I) {
        R.FreeLists[I].Free.clear();
        R.FreeLists[I].Quarantine.clear();
      }
      R.FreeLists.clear();
      R.Chunks.clear();
      R.Reservations.clear();
      R.NextChunk = R.MetadataBytes = 0;
      R.MetadataPoolSet = false;
    }

    VaRelease(kSpaceBeg, kSpaceUsed);
    Reserved = false;
    MetadataOk = false;
    Granule = MetaGranule = 0;
    QuarantinedBytes = 0;
    Count = {};
  }
  Initialized = false;
}

uptr Allocator::Allocate(hsa_amd_memory_pool_t Pool, uptr ClassId, uptr Size,
                         u32 AllocFlags, u32 Stack, u32 Tid, bool MapUser) {
  if (!Size || ClassId >= kNumSizeClasses)
    return 0;
  if (Size > ~uptr{0} - kMinRedzone)
    return 0;

  // Search for a chunk we can use in the allocator.
  const uptr Chunk = ClassIdToSize(ClassId);
  uptr Off = 0;
  uptr Bytes = Size;
  uptr RunAlign = Granule;
  if (MapUser) {
    if (Size > kSizeMask)
      return 0;
    const uptr Align = HeapAlign(Size);
    // This class holds the object in one chunk, redzone included.
    if (Chunk < Align)
      return 0;
    Off = LeadingRedzone(Size, Chunk, Align);
    if (Off + Size + kMinRedzone > Chunk)
      return 0;
    Bytes = Off + Size;
    RunAlign = Chunk;
  }

  const uptr AlignTo = AlignOf(ClassId, RunAlign);
  const uptr Stride = AlignTo / Chunk;
  const uptr MaxChunks = MaxUsableChunks(ClassId);
  if (!MaxChunks || Bytes > MaxChunks * Chunk)
    return 0;

  const uptr Need = RoundUpTo(Bytes, Chunk) / Chunk;
  if (!Need)
    return 0;

  // Find a free chunk in the current free list.
  const uptr List = FreeListFor(ClassId, Pool);
  uptr Begin = 0;
  if (!TakeFree(ClassId, List, Need, Stride, MapUser, &Begin) &&
      !Grow(Pool, ClassId, List, Need, Stride, MapUser, AllocFlags, &Begin)) {
    bool Got = false;
    while (Recycle(ClassId, List)) {
      if (TakeFree(ClassId, List, Need, Stride, MapUser, &Begin)) {
        Got = true;
        break;
      }
    }
    if (!Got) {
      VReport(2, "%s: no room in class %zu for %zu bytes\n", SanitizerToolName,
              ClassId, Size);
      return 0;
    }
  }

  if (MapUser && !MapPrefix(Pool, ClassId, Begin, Need, Bytes, AllocFlags)) {
    VReport(2, "%s: no backing for class %zu at chunk %zu\n", SanitizerToolName,
            ClassId, Begin);
    const bool Mapped = Regions[ClassId].Chunks[Begin].Mapped != 0;
    PushFree(ClassId, List, {Begin, Need, Mapped});
    return 0;
  }

  if (!MapUser) {
    Regions[ClassId].Reservations.push_back({Begin, Need, List, Size});
  } else {
    const Placement Place{Off, Size, Stack, Tid};
    MarkChunk(ClassId, Begin, List, Place);
    if (!Publish(ChunkBegin(ClassId, Begin) + Off, Size)) {
      ClearChunk(ClassId, Begin);
      const bool Mapped = Regions[ClassId].Chunks[Begin].Mapped != 0;
      PushFree(ClassId, List, {Begin, Need, Mapped});
      return 0;
    }
  }

  ++Count.Placed;
  if (!MapUser) {
    ++Count.CodeObjects;
    Count.CodeObjectChunks += Need;
  }

  const uptr Base = ChunkBegin(ClassId, Begin);
  const uptr At = Base + Off;
  VReport(3, "%s: placed %zu bytes at 0x%zx (class %zu, %zu chunks)\n",
          SanitizerToolName, Size, At, ClassId, Need);
  return At;
}

bool Allocator::Deallocate(uptr Addr, u32 Stack, u32 Tid) {
  if (!IsPlaced(Addr))
    return false;

  const uptr ClassId = GetSizeClass(Addr);
  const uptr Idx = GetChunkIdx(Addr, ClassId);
  Region& R = Regions[ClassId];
  if (Idx >= R.NextChunk)
    return false;

  ChunkState* S = Idx < R.Chunks.size() ? &R.Chunks[Idx] : nullptr;
  const uptr Base = S && S->RunLen ? ChunkBegin(ClassId, S->RunStart) : 0;
  if (!S || S->Gap || !S->Heap || !S->Live || !S->RunLen ||
      Addr != Base + S->UserOffset) {
    ReportInvalidFree(Addr, Stack, Tid);
    return true;
  }

  Quarantine(ClassId, S->List, {S->RunStart, S->RunLen, S->Mapped != 0}, Stack,
             Tid);
  ++Count.Freed;
  return true;
}

void Allocator::Release(uptr Addr) {
  if (!Reserved || !PointerIsMine(Addr))
    return;

  const uptr ClassId = GetSizeClass(Addr);
  const uptr Idx = GetChunkIdx(Addr, ClassId);
  Region& R = Regions[ClassId];
  if (Idx >= R.NextChunk)
    return;

  for (uptr I = 0; I < R.Reservations.size(); ++I) {
    Reservation& M = R.Reservations[I];
    if (M.Begin != Idx)
      continue;

    const uptr Len = M.Count;
    const uptr List = M.List;
    if (I + 1 != R.Reservations.size())
      R.Reservations[I] = R.Reservations.back();
    R.Reservations.pop_back();
    PushFree(ClassId, List, {Idx, Len, false});
    if (Count.CodeObjectChunks >= Len)
      Count.CodeObjectChunks -= Len;
    if (Count.CodeObjects)
      --Count.CodeObjects;
    return;
  }
}

bool Allocator::getAllocInfo(uptr Addr, AllocInfo* Out) const {
  if (!IsPlaced(Addr))
    return false;

  const uptr ClassId = GetSizeClass(Addr);
  const uptr Idx = GetChunkIdx(Addr, ClassId);
  const Region& R = Regions[ClassId];
  if (Idx >= R.NextChunk)
    return false;

  if (Idx < R.Chunks.size()) {
    const ChunkState& S = R.Chunks[Idx];
    if (!S.Gap && S.RunLen) {
      Out->Beg = ChunkBegin(ClassId, S.RunStart) + S.UserOffset;
      Out->Size = S.UserSize;
      Out->Pool = R.FreeLists[S.List].Pool;
      Out->Freed = !S.Live;
      Out->AllocStack = S.AllocStack;
      Out->FreeStack = S.FreeStack;
      Out->AllocTid = S.AllocTid;
      Out->FreeTid = S.FreeTid;
      return S.UserSize != 0;
    }
  }

  for (const Reservation& M : R.Reservations) {
    if (M.Begin != Idx || !M.UserSize)
      continue;
    Out->Beg = ChunkBegin(ClassId, Idx);
    Out->Size = M.UserSize;
    Out->Pool = R.FreeLists[M.List].Pool;
    Out->Freed = false;
    Out->AllocStack = Out->FreeStack = 0;
    Out->AllocTid = Out->FreeTid = 0;
    return true;
  }
  return false;
}

bool Allocator::AllowAccess(uptr Addr, const hsa_agent_t* Agents, u32 N) {
  if (!IsPlaced(Addr))
    return false;

  const uptr ClassId = GetSizeClass(Addr);
  const uptr Idx = GetChunkIdx(Addr, ClassId);
  const Region& R = Regions[ClassId];
  uptr UserBeg = Addr;
  uptr UserSize = 1;
  uptr Meta = GetMetadata(Addr, Idx);
  uptr MetaBytes = kMetadataSize;
  if (Idx < R.NextChunk && Idx < R.Chunks.size() && !R.Chunks[Idx].Gap &&
      R.Chunks[Idx].Heap && R.Chunks[Idx].RunLen && R.Chunks[Idx].UserSize) {
    const ChunkState& S = R.Chunks[Idx];
    UserBeg = ChunkBegin(ClassId, S.RunStart);
    UserSize = S.RunLen * ClassIdToSize(ClassId);
    Meta = GetMetadata(RegionBeg(ClassId), S.RunStart + S.RunLen - 1);
    MetaBytes = S.RunLen * kMetadataSize;
  } else if (Idx < R.NextChunk) {
    for (const Reservation& M : R.Reservations) {
      if (Idx < M.Begin || Idx - M.Begin >= M.Count)
        continue;
      UserBeg = ChunkBegin(ClassId, M.Begin);
      UserSize = M.Count * ClassIdToSize(ClassId);
      Meta = GetMetadata(RegionBeg(ClassId), M.Begin + M.Count - 1);
      MetaBytes = M.Count * kMetadataSize;
      break;
    }
  }
  if (!VaAllowAgents(UserBeg, UserSize, Agents, N))
    return false;
  return VaAllowAgents(Meta, MetaBytes, Agents, N);
}

static Allocator Instance;
Allocator& GetAllocator() { return Instance; }

}  // namespace __dasan
