//===-- scudo_allocator.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// Scudo Hardened Allocator implementation.
/// It uses the sanitizer_common allocator as a base and aims at mitigating
/// heap corruption vulnerabilities. It provides a checksum-guarded chunk
/// header, a delayed free list, and additional sanity checks.
///
//===----------------------------------------------------------------------===//

#include "scudo_allocator.h"
#include "scudo_crc32.h"
#include "scudo_tls.h"
#include "scudo_utils.h"

#include "sanitizer_common/sanitizer_allocator_interface.h"
#include "sanitizer_common/sanitizer_quarantine.h"

#include <limits.h>
#include <pthread.h>
#include <string.h>

namespace __scudo {

// Global static cookie, initialized at start-up.
static uptr Cookie;

// We default to software CRC32 if the alternatives are not supported, either
// at compilation or at runtime.
static atomic_uint8_t HashAlgorithm = { CRC32Software };

INLINE u32 computeCRC32(uptr Crc, uptr Value, uptr *Array, uptr ArraySize) {
  // If the hardware CRC32 feature is defined here, it was enabled everywhere,
  // as opposed to only for scudo_crc32.cpp. This means that other hardware
  // specific instructions were likely emitted at other places, and as a
  // result there is no reason to not use it here.
#if defined(__SSE4_2__) || defined(__ARM_FEATURE_CRC32)
  Crc = CRC32_INTRINSIC(Crc, Value);
  for (uptr i = 0; i < ArraySize; i++)
    Crc = CRC32_INTRINSIC(Crc, Array[i]);
  return Crc;
#else
  if (atomic_load_relaxed(&HashAlgorithm) == CRC32Hardware) {
    Crc = computeHardwareCRC32(Crc, Value);
    for (uptr i = 0; i < ArraySize; i++)
      Crc = computeHardwareCRC32(Crc, Array[i]);
    return Crc;
  }
  Crc = computeSoftwareCRC32(Crc, Value);
  for (uptr i = 0; i < ArraySize; i++)
    Crc = computeSoftwareCRC32(Crc, Array[i]);
  return Crc;
#endif  // defined(__SSE4_2__) || defined(__ARM_FEATURE_CRC32)
}

static ScudoBackendAllocator &getBackendAllocator();

struct ScudoChunk : UnpackedHeader {
  // We can't use the offset member of the chunk itself, as we would double
  // fetch it without any warranty that it wouldn't have been tampered. To
  // prevent this, we work with a local copy of the header.
  void *getAllocBeg(UnpackedHeader *Header) {
    return reinterpret_cast<void *>(
        reinterpret_cast<uptr>(this) - (Header->Offset << MinAlignmentLog));
  }

  // Returns the usable size for a chunk, meaning the amount of bytes from the
  // beginning of the user data to the end of the backend allocated chunk.
  uptr getUsableSize(UnpackedHeader *Header) {
    uptr Size =
        getBackendAllocator().GetActuallyAllocatedSize(getAllocBeg(Header),
                                                       Header->FromPrimary);
    if (Size == 0)
      return 0;
    return Size - AlignedChunkHeaderSize - (Header->Offset << MinAlignmentLog);
  }

  // Compute the checksum of the Chunk pointer and its ChunkHeader.
  u16 computeChecksum(UnpackedHeader *Header) const {
    UnpackedHeader ZeroChecksumHeader = *Header;
    ZeroChecksumHeader.Checksum = 0;
    uptr HeaderHolder[sizeof(UnpackedHeader) / sizeof(uptr)];
    memcpy(&HeaderHolder, &ZeroChecksumHeader, sizeof(HeaderHolder));
    u32 Crc = computeCRC32(Cookie, reinterpret_cast<uptr>(this), HeaderHolder,
                           ARRAY_SIZE(HeaderHolder));
    return static_cast<u16>(Crc);
  }

  // Checks the validity of a chunk by verifying its checksum. It doesn't
  // incur termination in the event of an invalid chunk.
  bool isValid() {
    UnpackedHeader NewUnpackedHeader;
    const AtomicPackedHeader *AtomicHeader =
        reinterpret_cast<const AtomicPackedHeader *>(this);
    PackedHeader NewPackedHeader = atomic_load_relaxed(AtomicHeader);
    NewUnpackedHeader = bit_cast<UnpackedHeader>(NewPackedHeader);
    return (NewUnpackedHeader.Checksum == computeChecksum(&NewUnpackedHeader));
  }

  // Nulls out a chunk header. When returning the chunk to the backend, there
  // is no need to store a valid ChunkAvailable header, as this would be
  // computationally expensive. Zeroing out serves the same purpose by making
  // the header invalid. In the extremely rare event where 0 would be a valid
  // checksum for the chunk, the state of the chunk is ChunkAvailable anyway.
  COMPILER_CHECK(ChunkAvailable == 0);
  void eraseHeader() {
    PackedHeader NullPackedHeader = 0;
    AtomicPackedHeader *AtomicHeader =
        reinterpret_cast<AtomicPackedHeader *>(this);
    atomic_store_relaxed(AtomicHeader, NullPackedHeader);
  }

  // Loads and unpacks the header, verifying the checksum in the process.
  void loadHeader(UnpackedHeader *NewUnpackedHeader) const {
    const AtomicPackedHeader *AtomicHeader =
        reinterpret_cast<const AtomicPackedHeader *>(this);
    PackedHeader NewPackedHeader = atomic_load_relaxed(AtomicHeader);
    *NewUnpackedHeader = bit_cast<UnpackedHeader>(NewPackedHeader);
    if (UNLIKELY(NewUnpackedHeader->Checksum !=
        computeChecksum(NewUnpackedHeader))) {
      dieWithMessage("ERROR: corrupted chunk header at address %p\n", this);
    }
  }

  // Packs and stores the header, computing the checksum in the process.
  void storeHeader(UnpackedHeader *NewUnpackedHeader) {
    NewUnpackedHeader->Checksum = computeChecksum(NewUnpackedHeader);
    PackedHeader NewPackedHeader = bit_cast<PackedHeader>(*NewUnpackedHeader);
    AtomicPackedHeader *AtomicHeader =
        reinterpret_cast<AtomicPackedHeader *>(this);
    atomic_store_relaxed(AtomicHeader, NewPackedHeader);
  }

  // Packs and stores the header, computing the checksum in the process. We
  // compare the current header with the expected provided one to ensure that
  // we are not being raced by a corruption occurring in another thread.
  void compareExchangeHeader(UnpackedHeader *NewUnpackedHeader,
                             UnpackedHeader *OldUnpackedHeader) {
    NewUnpackedHeader->Checksum = computeChecksum(NewUnpackedHeader);
    PackedHeader NewPackedHeader = bit_cast<PackedHeader>(*NewUnpackedHeader);
    PackedHeader OldPackedHeader = bit_cast<PackedHeader>(*OldUnpackedHeader);
    AtomicPackedHeader *AtomicHeader =
        reinterpret_cast<AtomicPackedHeader *>(this);
    if (UNLIKELY(!atomic_compare_exchange_strong(AtomicHeader,
                                                 &OldPackedHeader,
                                                 NewPackedHeader,
                                                 memory_order_relaxed))) {
      dieWithMessage("ERROR: race on chunk header at address %p\n", this);
    }
  }
};

ScudoChunk *getScudoChunk(uptr UserBeg) {
  return reinterpret_cast<ScudoChunk *>(UserBeg - AlignedChunkHeaderSize);
}

struct AllocatorOptions {
  u32 QuarantineSizeMb;
  u32 ThreadLocalQuarantineSizeKb;
  bool MayReturnNull;
  s32 ReleaseToOSIntervalMs;
  bool DeallocationTypeMismatch;
  bool DeleteSizeMismatch;
  bool ZeroContents;

  void setFrom(const Flags *f, const CommonFlags *cf);
  void copyTo(Flags *f, CommonFlags *cf) const;
};

void AllocatorOptions::setFrom(const Flags *f, const CommonFlags *cf) {
  MayReturnNull = cf->allocator_may_return_null;
  ReleaseToOSIntervalMs = cf->allocator_release_to_os_interval_ms;
  QuarantineSizeMb = f->QuarantineSizeMb;
  ThreadLocalQuarantineSizeKb = f->ThreadLocalQuarantineSizeKb;
  DeallocationTypeMismatch = f->DeallocationTypeMismatch;
  DeleteSizeMismatch = f->DeleteSizeMismatch;
  ZeroContents = f->ZeroContents;
}

void AllocatorOptions::copyTo(Flags *f, CommonFlags *cf) const {
  cf->allocator_may_return_null = MayReturnNull;
  cf->allocator_release_to_os_interval_ms = ReleaseToOSIntervalMs;
  f->QuarantineSizeMb = QuarantineSizeMb;
  f->ThreadLocalQuarantineSizeKb = ThreadLocalQuarantineSizeKb;
  f->DeallocationTypeMismatch = DeallocationTypeMismatch;
  f->DeleteSizeMismatch = DeleteSizeMismatch;
  f->ZeroContents = ZeroContents;
}

static void initScudoInternal(const AllocatorOptions &Options);

static bool ScudoInitIsRunning = false;

void initScudo() {
  SanitizerToolName = "Scudo";
  CHECK(!ScudoInitIsRunning && "Scudo init calls itself!");
  ScudoInitIsRunning = true;

  // Check if hardware CRC32 is supported in the binary and by the platform, if
  // so, opt for the CRC32 hardware version of the checksum.
  if (computeHardwareCRC32 && testCPUFeature(CRC32CPUFeature))
    atomic_store_relaxed(&HashAlgorithm, CRC32Hardware);

  initFlags();

  AllocatorOptions Options;
  Options.setFrom(getFlags(), common_flags());
  initScudoInternal(Options);

  // TODO(kostyak): determine if MaybeStartBackgroudThread could be of some use.

  ScudoInitIsRunning = false;
}

struct QuarantineCallback {
  explicit QuarantineCallback(AllocatorCache *Cache)
    : Cache_(Cache) {}

  // Chunk recycling function, returns a quarantined chunk to the backend,
  // first making sure it hasn't been tampered with.
  void Recycle(ScudoChunk *Chunk) {
    UnpackedHeader Header;
    Chunk->loadHeader(&Header);
    if (UNLIKELY(Header.State != ChunkQuarantine)) {
      dieWithMessage("ERROR: invalid chunk state when recycling address %p\n",
                     Chunk);
    }
    Chunk->eraseHeader();
    void *Ptr = Chunk->getAllocBeg(&Header);
    getBackendAllocator().Deallocate(Cache_, Ptr, Header.FromPrimary);
  }

  // Internal quarantine allocation and deallocation functions. We first check
  // that the batches are indeed serviced by the Primary.
  // TODO(kostyak): figure out the best way to protect the batches.
  COMPILER_CHECK(sizeof(QuarantineBatch) < SizeClassMap::kMaxSize);
  void *Allocate(uptr Size) {
    return getBackendAllocator().Allocate(Cache_, Size, MinAlignment, true);
  }

  void Deallocate(void *Ptr) {
    getBackendAllocator().Deallocate(Cache_, Ptr, true);
  }

  AllocatorCache *Cache_;
};

typedef Quarantine<QuarantineCallback, ScudoChunk> ScudoQuarantine;
typedef ScudoQuarantine::Cache ScudoQuarantineCache;
COMPILER_CHECK(sizeof(ScudoQuarantineCache) <=
               sizeof(ScudoThreadContext::QuarantineCachePlaceHolder));

AllocatorCache *getAllocatorCache(ScudoThreadContext *ThreadContext) {
  return &ThreadContext->Cache;
}

ScudoQuarantineCache *getQuarantineCache(ScudoThreadContext *ThreadContext) {
  return reinterpret_cast<
      ScudoQuarantineCache *>(ThreadContext->QuarantineCachePlaceHolder);
}

Xorshift128Plus *getPrng(ScudoThreadContext *ThreadContext) {
  return &ThreadContext->Prng;
}

struct ScudoAllocator {
  static const uptr MaxAllowedMallocSize =
      FIRST_32_SECOND_64(2UL << 30, 1ULL << 40);

  ScudoBackendAllocator BackendAllocator;
  ScudoQuarantine AllocatorQuarantine;

  // The fallback caches are used when the thread local caches have been
  // 'detroyed' on thread tear-down. They are protected by a Mutex as they can
  // be accessed by different threads.
  StaticSpinMutex FallbackMutex;
  AllocatorCache FallbackAllocatorCache;
  ScudoQuarantineCache FallbackQuarantineCache;
  Xorshift128Plus FallbackPrng;

  bool DeallocationTypeMismatch;
  bool ZeroContents;
  bool DeleteSizeMismatch;

  explicit ScudoAllocator(LinkerInitialized)
    : AllocatorQuarantine(LINKER_INITIALIZED),
      FallbackQuarantineCache(LINKER_INITIALIZED) {}

  void init(const AllocatorOptions &Options) {
    // Verify that the header offset field can hold the maximum offset. In the
    // case of the Secondary allocator, it takes care of alignment and the
    // offset will always be 0. In the case of the Primary, the worst case
    // scenario happens in the last size class, when the backend allocation
    // would already be aligned on the requested alignment, which would happen
    // to be the maximum alignment that would fit in that size class. As a
    // result, the maximum offset will be at most the maximum alignment for the
    // last size class minus the header size, in multiples of MinAlignment.
    UnpackedHeader Header = {};
    uptr MaxPrimaryAlignment = 1 << MostSignificantSetBitIndex(
        SizeClassMap::kMaxSize - MinAlignment);
    uptr MaxOffset = (MaxPrimaryAlignment - AlignedChunkHeaderSize) >>
        MinAlignmentLog;
    Header.Offset = MaxOffset;
    if (Header.Offset != MaxOffset) {
      dieWithMessage("ERROR: the maximum possible offset doesn't fit in the "
                     "header\n");
    }
    // Verify that we can fit the maximum size or amount of unused bytes in the
    // header. Given that the Secondary fits the allocation to a page, the worst
    // case scenario happens in the Primary. It will depend on the second to
    // last and last class sizes, as well as the dynamic base for the Primary.
    // The following is an over-approximation that works for our needs.
    uptr MaxSizeOrUnusedBytes = SizeClassMap::kMaxSize - 1;
    Header.SizeOrUnusedBytes = MaxSizeOrUnusedBytes;
    if (Header.SizeOrUnusedBytes != MaxSizeOrUnusedBytes) {
      dieWithMessage("ERROR: the maximum possible unused bytes doesn't fit in "
                     "the header\n");
    }

    DeallocationTypeMismatch = Options.DeallocationTypeMismatch;
    DeleteSizeMismatch = Options.DeleteSizeMismatch;
    ZeroContents = Options.ZeroContents;
    BackendAllocator.Init(Options.MayReturnNull, Options.ReleaseToOSIntervalMs);
    AllocatorQuarantine.Init(
        static_cast<uptr>(Options.QuarantineSizeMb) << 20,
        static_cast<uptr>(Options.ThreadLocalQuarantineSizeKb) << 10);
    BackendAllocator.InitCache(&FallbackAllocatorCache);
    FallbackPrng.initFromURandom();
    Cookie = FallbackPrng.getNext();
  }

  // Helper function that checks for a valid Scudo chunk. nullptr isn't.
  bool isValidPointer(const void *UserPtr) {
    initThreadMaybe();
    if (!UserPtr)
      return false;
    uptr UserBeg = reinterpret_cast<uptr>(UserPtr);
    if (!IsAligned(UserBeg, MinAlignment))
      return false;
    return getScudoChunk(UserBeg)->isValid();
  }

  // Allocates a chunk.
  void *allocate(uptr Size, uptr Alignment, AllocType Type,
                 bool ForceZeroContents = false) {
    initThreadMaybe();
    if (UNLIKELY(!IsPowerOfTwo(Alignment))) {
      dieWithMessage("ERROR: alignment is not a power of 2\n");
    }
    if (Alignment > MaxAlignment)
      return BackendAllocator.ReturnNullOrDieOnBadRequest();
    if (Alignment < MinAlignment)
      Alignment = MinAlignment;
    if (Size >= MaxAllowedMallocSize)
      return BackendAllocator.ReturnNullOrDieOnBadRequest();
    if (Size == 0)
      Size = 1;

    uptr NeededSize = RoundUpTo(Size, MinAlignment) + AlignedChunkHeaderSize;
    uptr AlignedSize = (Alignment > MinAlignment) ?
        NeededSize + (Alignment - AlignedChunkHeaderSize) : NeededSize;
    if (AlignedSize >= MaxAllowedMallocSize)
      return BackendAllocator.ReturnNullOrDieOnBadRequest();

    // Primary and Secondary backed allocations have a different treatment. We
    // deal with alignment requirements of Primary serviced allocations here,
    // but the Secondary will take care of its own alignment needs.
    bool FromPrimary = PrimaryAllocator::CanAllocate(AlignedSize, MinAlignment);

    void *Ptr;
    uptr Salt;
    uptr AllocationSize = FromPrimary ? AlignedSize : NeededSize;
    uptr AllocationAlignment = FromPrimary ? MinAlignment : Alignment;
    ScudoThreadContext *ThreadContext = getThreadContextAndLock();
    if (LIKELY(ThreadContext)) {
      Salt = getPrng(ThreadContext)->getNext();
      Ptr = BackendAllocator.Allocate(getAllocatorCache(ThreadContext),
                                      AllocationSize, AllocationAlignment,
                                      FromPrimary);
      ThreadContext->unlock();
    } else {
      SpinMutexLock l(&FallbackMutex);
      Salt = FallbackPrng.getNext();
      Ptr = BackendAllocator.Allocate(&FallbackAllocatorCache, AllocationSize,
                                      AllocationAlignment, FromPrimary);
    }
    if (!Ptr)
      return BackendAllocator.ReturnNullOrDieOnOOM();

    // If requested, we will zero out the entire contents of the returned chunk.
    if ((ForceZeroContents || ZeroContents) && FromPrimary)
       memset(Ptr, 0,
              BackendAllocator.GetActuallyAllocatedSize(Ptr, FromPrimary));

    UnpackedHeader Header = {};
    uptr AllocBeg = reinterpret_cast<uptr>(Ptr);
    uptr UserBeg = AllocBeg + AlignedChunkHeaderSize;
    if (!IsAligned(UserBeg, Alignment)) {
      // Since the Secondary takes care of alignment, a non-aligned pointer
      // means it is from the Primary. It is also the only case where the offset
      // field of the header would be non-zero.
      CHECK(FromPrimary);
      UserBeg = RoundUpTo(UserBeg, Alignment);
      uptr Offset = UserBeg - AlignedChunkHeaderSize - AllocBeg;
      Header.Offset = Offset >> MinAlignmentLog;
    }
    CHECK_LE(UserBeg + Size, AllocBeg + AllocationSize);
    Header.State = ChunkAllocated;
    Header.AllocType = Type;
    if (FromPrimary) {
      Header.FromPrimary = FromPrimary;
      Header.SizeOrUnusedBytes = Size;
    } else {
      // The secondary fits the allocations to a page, so the amount of unused
      // bytes is the difference between the end of the user allocation and the
      // next page boundary.
      uptr PageSize = GetPageSizeCached();
      uptr TrailingBytes = (UserBeg + Size) & (PageSize - 1);
      if (TrailingBytes)
        Header.SizeOrUnusedBytes = PageSize - TrailingBytes;
    }
    Header.Salt = static_cast<u8>(Salt);
    getScudoChunk(UserBeg)->storeHeader(&Header);
    void *UserPtr = reinterpret_cast<void *>(UserBeg);
    // if (&__sanitizer_malloc_hook) __sanitizer_malloc_hook(UserPtr, Size);
    return UserPtr;
  }

  // Place a chunk in the quarantine. In the event of a zero-sized quarantine,
  // we directly deallocate the chunk, otherwise the flow would lead to the
  // chunk being loaded (and checked) twice, and stored (and checksummed) once,
  // with no additional security value.
  void quarantineOrDeallocateChunk(ScudoChunk *Chunk, UnpackedHeader *Header,
                                   uptr Size) {
    bool FromPrimary = Header->FromPrimary;
    bool BypassQuarantine = (AllocatorQuarantine.GetCacheSize() == 0);
    if (BypassQuarantine) {
      Chunk->eraseHeader();
      void *Ptr = Chunk->getAllocBeg(Header);
      ScudoThreadContext *ThreadContext = getThreadContextAndLock();
      if (LIKELY(ThreadContext)) {
        getBackendAllocator().Deallocate(getAllocatorCache(ThreadContext), Ptr,
                                         FromPrimary);
        ThreadContext->unlock();
      } else {
        SpinMutexLock Lock(&FallbackMutex);
        getBackendAllocator().Deallocate(&FallbackAllocatorCache, Ptr,
                                         FromPrimary);
      }
    } else {
      UnpackedHeader NewHeader = *Header;
      NewHeader.State = ChunkQuarantine;
      Chunk->compareExchangeHeader(&NewHeader, Header);
      ScudoThreadContext *ThreadContext = getThreadContextAndLock();
      if (LIKELY(ThreadContext)) {
        AllocatorQuarantine.Put(getQuarantineCache(ThreadContext),
                                QuarantineCallback(
                                    getAllocatorCache(ThreadContext)),
                                Chunk, Size);
        ThreadContext->unlock();
      } else {
        SpinMutexLock l(&FallbackMutex);
        AllocatorQuarantine.Put(&FallbackQuarantineCache,
                                QuarantineCallback(&FallbackAllocatorCache),
                                Chunk, Size);
      }
    }
  }

  // Deallocates a Chunk, which means adding it to the delayed free list (or
  // Quarantine).
  void deallocate(void *UserPtr, uptr DeleteSize, AllocType Type) {
    initThreadMaybe();
    // if (&__sanitizer_free_hook) __sanitizer_free_hook(UserPtr);
    if (!UserPtr)
      return;
    uptr UserBeg = reinterpret_cast<uptr>(UserPtr);
    if (UNLIKELY(!IsAligned(UserBeg, MinAlignment))) {
      dieWithMessage("ERROR: attempted to deallocate a chunk not properly "
                     "aligned at address %p\n", UserPtr);
    }
    ScudoChunk *Chunk = getScudoChunk(UserBeg);
    UnpackedHeader OldHeader;
    Chunk->loadHeader(&OldHeader);
    if (UNLIKELY(OldHeader.State != ChunkAllocated)) {
      dieWithMessage("ERROR: invalid chunk state when deallocating address "
                     "%p\n", UserPtr);
    }
    if (DeallocationTypeMismatch) {
      // The deallocation type has to match the allocation one.
      if (OldHeader.AllocType != Type) {
        // With the exception of memalign'd Chunks, that can be still be free'd.
        if (OldHeader.AllocType != FromMemalign || Type != FromMalloc) {
          dieWithMessage("ERROR: allocation type mismatch on address %p\n",
                         UserPtr);
        }
      }
    }
    uptr Size = OldHeader.FromPrimary ? OldHeader.SizeOrUnusedBytes :
        Chunk->getUsableSize(&OldHeader) - OldHeader.SizeOrUnusedBytes;
    if (DeleteSizeMismatch) {
      if (DeleteSize && DeleteSize != Size) {
        dieWithMessage("ERROR: invalid sized delete on chunk at address %p\n",
                       UserPtr);
      }
    }

    // If a small memory amount was allocated with a larger alignment, we want
    // to take that into account. Otherwise the Quarantine would be filled with
    // tiny chunks, taking a lot of VA memory. This is an approximation of the
    // usable size, that allows us to not call GetActuallyAllocatedSize.
    uptr LiableSize = Size + (OldHeader.Offset << MinAlignment);
    quarantineOrDeallocateChunk(Chunk, &OldHeader, LiableSize);
  }

  // Reallocates a chunk. We can save on a new allocation if the new requested
  // size still fits in the chunk.
  void *reallocate(void *OldPtr, uptr NewSize) {
    initThreadMaybe();
    uptr UserBeg = reinterpret_cast<uptr>(OldPtr);
    if (UNLIKELY(!IsAligned(UserBeg, MinAlignment))) {
      dieWithMessage("ERROR: attempted to reallocate a chunk not properly "
                     "aligned at address %p\n", OldPtr);
    }
    ScudoChunk *Chunk = getScudoChunk(UserBeg);
    UnpackedHeader OldHeader;
    Chunk->loadHeader(&OldHeader);
    if (UNLIKELY(OldHeader.State != ChunkAllocated)) {
      dieWithMessage("ERROR: invalid chunk state when reallocating address "
                     "%p\n", OldPtr);
    }
    if (UNLIKELY(OldHeader.AllocType != FromMalloc)) {
      dieWithMessage("ERROR: invalid chunk type when reallocating address %p\n",
                     OldPtr);
    }
    uptr UsableSize = Chunk->getUsableSize(&OldHeader);
    // The new size still fits in the current chunk, and the size difference
    // is reasonable.
    if (NewSize <= UsableSize &&
        (UsableSize - NewSize) < (SizeClassMap::kMaxSize / 2)) {
      UnpackedHeader NewHeader = OldHeader;
      NewHeader.SizeOrUnusedBytes =
                OldHeader.FromPrimary ? NewSize : UsableSize - NewSize;
      Chunk->compareExchangeHeader(&NewHeader, &OldHeader);
      return OldPtr;
    }
    // Otherwise, we have to allocate a new chunk and copy the contents of the
    // old one.
    void *NewPtr = allocate(NewSize, MinAlignment, FromMalloc);
    if (NewPtr) {
      uptr OldSize = OldHeader.FromPrimary ? OldHeader.SizeOrUnusedBytes :
          UsableSize - OldHeader.SizeOrUnusedBytes;
      memcpy(NewPtr, OldPtr, Min(NewSize, OldSize));
      quarantineOrDeallocateChunk(Chunk, &OldHeader, UsableSize);
    }
    return NewPtr;
  }

  // Helper function that returns the actual usable size of a chunk.
  uptr getUsableSize(const void *Ptr) {
    initThreadMaybe();
    if (!Ptr)
      return 0;
    uptr UserBeg = reinterpret_cast<uptr>(Ptr);
    ScudoChunk *Chunk = getScudoChunk(UserBeg);
    UnpackedHeader Header;
    Chunk->loadHeader(&Header);
    // Getting the usable size of a chunk only makes sense if it's allocated.
    if (UNLIKELY(Header.State != ChunkAllocated)) {
      dieWithMessage("ERROR: invalid chunk state when sizing address %p\n",
                     Ptr);
    }
    return Chunk->getUsableSize(&Header);
  }

  void *calloc(uptr NMemB, uptr Size) {
    initThreadMaybe();
    uptr Total = NMemB * Size;
    if (Size != 0 && Total / Size != NMemB)  // Overflow check
      return BackendAllocator.ReturnNullOrDieOnBadRequest();
    return allocate(Total, MinAlignment, FromMalloc, true);
  }

  void commitBack(ScudoThreadContext *ThreadContext) {
    AllocatorCache *Cache = getAllocatorCache(ThreadContext);
    AllocatorQuarantine.Drain(getQuarantineCache(ThreadContext),
                              QuarantineCallback(Cache));
    BackendAllocator.DestroyCache(Cache);
  }

  uptr getStats(AllocatorStat StatType) {
    initThreadMaybe();
    uptr stats[AllocatorStatCount];
    BackendAllocator.GetStats(stats);
    return stats[StatType];
  }
};

static ScudoAllocator Instance(LINKER_INITIALIZED);

static ScudoBackendAllocator &getBackendAllocator() {
  return Instance.BackendAllocator;
}

static void initScudoInternal(const AllocatorOptions &Options) {
  Instance.init(Options);
}

void ScudoThreadContext::init() {
  getBackendAllocator().InitCache(&Cache);
  Prng.initFromURandom();
  memset(QuarantineCachePlaceHolder, 0, sizeof(QuarantineCachePlaceHolder));
}

void ScudoThreadContext::commitBack() {
  Instance.commitBack(this);
}

void *scudoMalloc(uptr Size, AllocType Type) {
  return Instance.allocate(Size, MinAlignment, Type);
}

void scudoFree(void *Ptr, AllocType Type) {
  Instance.deallocate(Ptr, 0, Type);
}

void scudoSizedFree(void *Ptr, uptr Size, AllocType Type) {
  Instance.deallocate(Ptr, Size, Type);
}

void *scudoRealloc(void *Ptr, uptr Size) {
  if (!Ptr)
    return Instance.allocate(Size, MinAlignment, FromMalloc);
  if (Size == 0) {
    Instance.deallocate(Ptr, 0, FromMalloc);
    return nullptr;
  }
  return Instance.reallocate(Ptr, Size);
}

void *scudoCalloc(uptr NMemB, uptr Size) {
  return Instance.calloc(NMemB, Size);
}

void *scudoValloc(uptr Size) {
  return Instance.allocate(Size, GetPageSizeCached(), FromMemalign);
}

void *scudoMemalign(uptr Alignment, uptr Size) {
  return Instance.allocate(Size, Alignment, FromMemalign);
}

void *scudoPvalloc(uptr Size) {
  uptr PageSize = GetPageSizeCached();
  Size = RoundUpTo(Size, PageSize);
  if (Size == 0) {
    // pvalloc(0) should allocate one page.
    Size = PageSize;
  }
  return Instance.allocate(Size, PageSize, FromMemalign);
}

int scudoPosixMemalign(void **MemPtr, uptr Alignment, uptr Size) {
  *MemPtr = Instance.allocate(Size, Alignment, FromMemalign);
  return 0;
}

void *scudoAlignedAlloc(uptr Alignment, uptr Size) {
  // size must be a multiple of the alignment. To avoid a division, we first
  // make sure that alignment is a power of 2.
  CHECK(IsPowerOfTwo(Alignment));
  CHECK_EQ((Size & (Alignment - 1)), 0);
  return Instance.allocate(Size, Alignment, FromMalloc);
}

uptr scudoMallocUsableSize(void *Ptr) {
  return Instance.getUsableSize(Ptr);
}

}  // namespace __scudo

using namespace __scudo;

// MallocExtension helper functions

uptr __sanitizer_get_current_allocated_bytes() {
  return Instance.getStats(AllocatorStatAllocated);
}

uptr __sanitizer_get_heap_size() {
  return Instance.getStats(AllocatorStatMapped);
}

uptr __sanitizer_get_free_bytes() {
  return 1;
}

uptr __sanitizer_get_unmapped_bytes() {
  return 1;
}

uptr __sanitizer_get_estimated_allocated_size(uptr size) {
  return size;
}

int __sanitizer_get_ownership(const void *Ptr) {
  return Instance.isValidPointer(Ptr);
}

uptr __sanitizer_get_allocated_size(const void *Ptr) {
  return Instance.getUsableSize(Ptr);
}
