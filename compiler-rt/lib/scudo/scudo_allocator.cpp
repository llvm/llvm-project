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
#include "scudo_utils.h"
#include "scudo_allocator_secondary.h"

#include "sanitizer_common/sanitizer_allocator_interface.h"
#include "sanitizer_common/sanitizer_quarantine.h"

#include <limits.h>
#include <pthread.h>
#include <smmintrin.h>

#include <atomic>
#include <cstring>

namespace __scudo {

const uptr MinAlignmentLog = 4; // 16 bytes for x64
const uptr MaxAlignmentLog = 24;

struct AP {
  static const uptr kSpaceBeg = ~0ULL;
  static const uptr kSpaceSize = 0x10000000000ULL;
  static const uptr kMetadataSize = 0;
  typedef DefaultSizeClassMap SizeClassMap;
  typedef NoOpMapUnmapCallback MapUnmapCallback;
  static const uptr kFlags =
      SizeClassAllocator64FlagMasks::kRandomShuffleChunks;
};

typedef SizeClassAllocator64<AP> PrimaryAllocator;
typedef SizeClassAllocatorLocalCache<PrimaryAllocator> AllocatorCache;
typedef ScudoLargeMmapAllocator SecondaryAllocator;
typedef CombinedAllocator<PrimaryAllocator, AllocatorCache, SecondaryAllocator>
  ScudoAllocator;

static ScudoAllocator &getAllocator();

static thread_local Xorshift128Plus Prng;
// Global static cookie, initialized at start-up.
static u64 Cookie;

enum ChunkState : u8 {
  ChunkAvailable  = 0,
  ChunkAllocated  = 1,
  ChunkQuarantine = 2
};

typedef unsigned __int128 PackedHeader;
typedef std::atomic<PackedHeader> AtomicPackedHeader;

// Our header requires 128-bit of storage on x64 (the only platform supported
// as of now), which fits nicely with the alignment requirements.
// Having the offset saves us from using functions such as GetBlockBegin, that
// is fairly costly. Our first implementation used the MetaData as well, which
// offers the advantage of being stored away from the chunk itself, but
// accessing it was costly as well.
// The header will be atomically loaded and stored using the 16-byte primitives
// offered by the platform (likely requires cmpxchg16b support).
struct UnpackedHeader {
  // 1st 8 bytes
  u16 Checksum      : 16;
  u64 RequestedSize : 40; // Needed for reallocation purposes.
  u8  State         : 2;  // available, allocated, or quarantined
  u8  AllocType     : 2;  // malloc, new, new[], or memalign
  u8  Unused_0_     : 4;
  // 2nd 8 bytes
  u64 Offset        : 20; // Offset from the beginning of the backend
                          // allocation to the beginning chunk itself, in
                          // multiples of MinAlignment. See comment about its
                          // maximum value and test in init().
  u64 Unused_1_     : 28;
  u16 Salt          : 16;
};

COMPILER_CHECK(sizeof(UnpackedHeader) == sizeof(PackedHeader));

const uptr ChunkHeaderSize = sizeof(PackedHeader);

struct ScudoChunk : UnpackedHeader {
  // We can't use the offset member of the chunk itself, as we would double
  // fetch it without any warranty that it wouldn't have been tampered. To
  // prevent this, we work with a local copy of the header.
  void *AllocBeg(UnpackedHeader *Header) {
    return reinterpret_cast<void *>(
        reinterpret_cast<uptr>(this) - (Header->Offset << MinAlignmentLog));
  }

  // CRC32 checksum of the Chunk pointer and its ChunkHeader.
  // It currently uses the Intel Nehalem SSE4.2 crc32 64-bit instruction.
  u16 Checksum(UnpackedHeader *Header) const {
    u64 HeaderHolder[2];
    memcpy(HeaderHolder, Header, sizeof(HeaderHolder));
    u64 Crc = _mm_crc32_u64(Cookie, reinterpret_cast<uptr>(this));
    // This is somewhat of a shortcut. The checksum is stored in the 16 least
    // significant bits of the first 8 bytes of the header, hence zero-ing
    // those bits out. It would be more valid to zero the checksum field of the
    // UnpackedHeader, but would require holding an additional copy of it.
    Crc = _mm_crc32_u64(Crc, HeaderHolder[0] & 0xffffffffffff0000ULL);
    Crc = _mm_crc32_u64(Crc, HeaderHolder[1]);
    return static_cast<u16>(Crc);
  }

  // Loads and unpacks the header, verifying the checksum in the process.
  void loadHeader(UnpackedHeader *NewUnpackedHeader) const {
    const AtomicPackedHeader *AtomicHeader =
        reinterpret_cast<const AtomicPackedHeader *>(this);
    PackedHeader NewPackedHeader =
        AtomicHeader->load(std::memory_order_relaxed);
    *NewUnpackedHeader = bit_cast<UnpackedHeader>(NewPackedHeader);
    if ((NewUnpackedHeader->Unused_0_ != 0) ||
        (NewUnpackedHeader->Unused_1_ != 0) ||
        (NewUnpackedHeader->Checksum != Checksum(NewUnpackedHeader))) {
      dieWithMessage("ERROR: corrupted chunk header at address %p\n", this);
    }
  }

  // Packs and stores the header, computing the checksum in the process.
  void storeHeader(UnpackedHeader *NewUnpackedHeader) {
    NewUnpackedHeader->Checksum = Checksum(NewUnpackedHeader);
    PackedHeader NewPackedHeader = bit_cast<PackedHeader>(*NewUnpackedHeader);
    AtomicPackedHeader *AtomicHeader =
        reinterpret_cast<AtomicPackedHeader *>(this);
    AtomicHeader->store(NewPackedHeader, std::memory_order_relaxed);
  }

  // Packs and stores the header, computing the checksum in the process. We
  // compare the current header with the expected provided one to ensure that
  // we are not being raced by a corruption occurring in another thread.
  void compareExchangeHeader(UnpackedHeader *NewUnpackedHeader,
                             UnpackedHeader *OldUnpackedHeader) {
    NewUnpackedHeader->Checksum = Checksum(NewUnpackedHeader);
    PackedHeader NewPackedHeader = bit_cast<PackedHeader>(*NewUnpackedHeader);
    PackedHeader OldPackedHeader = bit_cast<PackedHeader>(*OldUnpackedHeader);
    AtomicPackedHeader *AtomicHeader =
        reinterpret_cast<AtomicPackedHeader *>(this);
    if (!AtomicHeader->compare_exchange_strong(OldPackedHeader,
                                               NewPackedHeader,
                                               std::memory_order_relaxed,
                                               std::memory_order_relaxed)) {
      dieWithMessage("ERROR: race on chunk header at address %p\n", this);
    }
  }
};

static bool ScudoInitIsRunning = false;

static pthread_once_t GlobalInited = PTHREAD_ONCE_INIT;
static pthread_key_t pkey;

static thread_local bool ThreadInited = false;
static thread_local bool ThreadTornDown = false;
static thread_local AllocatorCache Cache;

static void teardownThread(void *p) {
  uptr v = reinterpret_cast<uptr>(p);
  // The glibc POSIX thread-local-storage deallocation routine calls user
  // provided destructors in a loop of PTHREAD_DESTRUCTOR_ITERATIONS.
  // We want to be called last since other destructors might call free and the
  // like, so we wait until PTHREAD_DESTRUCTOR_ITERATIONS before draining the
  // quarantine and swallowing the cache.
  if (v < PTHREAD_DESTRUCTOR_ITERATIONS) {
    pthread_setspecific(pkey, reinterpret_cast<void *>(v + 1));
    return;
  }
  drainQuarantine();
  getAllocator().DestroyCache(&Cache);
  ThreadTornDown = true;
}

static void initInternal() {
  SanitizerToolName = "Scudo";
  CHECK(!ScudoInitIsRunning && "Scudo init calls itself!");
  ScudoInitIsRunning = true;

  initFlags();

  AllocatorOptions Options;
  Options.setFrom(getFlags(), common_flags());
  initAllocator(Options);

  ScudoInitIsRunning = false;
}

static void initGlobal() {
  pthread_key_create(&pkey, teardownThread);
  initInternal();
}

static void NOINLINE initThread() {
  pthread_once(&GlobalInited, initGlobal);
  pthread_setspecific(pkey, reinterpret_cast<void *>(1));
  getAllocator().InitCache(&Cache);
  ThreadInited = true;
}

struct QuarantineCallback {
  explicit QuarantineCallback(AllocatorCache *Cache)
    : Cache_(Cache) {}

  // Chunk recycling function, returns a quarantined chunk to the backend.
  void Recycle(ScudoChunk *Chunk) {
    UnpackedHeader Header;
    Chunk->loadHeader(&Header);
    if (Header.State != ChunkQuarantine) {
      dieWithMessage("ERROR: invalid chunk state when recycling address %p\n",
                     Chunk);
    }
    void *Ptr = Chunk->AllocBeg(&Header);
    getAllocator().Deallocate(Cache_, Ptr);
  }

  /// Internal quarantine allocation and deallocation functions.
  void *Allocate(uptr Size) {
    // The internal quarantine memory cannot be protected by us. But the only
    // structures allocated are QuarantineBatch, that are 8KB for x64. So we
    // will use mmap for those, and given that Deallocate doesn't pass a size
    // in, we enforce the size of the allocation to be sizeof(QuarantineBatch).
    // TODO(kostyak): switching to mmap impacts greatly performances, we have
    //                to find another solution
    // CHECK_EQ(Size, sizeof(QuarantineBatch));
    // return MmapOrDie(Size, "QuarantineBatch");
    return getAllocator().Allocate(Cache_, Size, 1, false);
  }

  void Deallocate(void *Ptr) {
    // UnmapOrDie(Ptr, sizeof(QuarantineBatch));
    getAllocator().Deallocate(Cache_, Ptr);
  }

  AllocatorCache *Cache_;
};

typedef Quarantine<QuarantineCallback, ScudoChunk> ScudoQuarantine;
typedef ScudoQuarantine::Cache QuarantineCache;
static thread_local QuarantineCache ThreadQuarantineCache;

void AllocatorOptions::setFrom(const Flags *f, const CommonFlags *cf) {
  MayReturnNull = cf->allocator_may_return_null;
  QuarantineSizeMb = f->QuarantineSizeMb;
  ThreadLocalQuarantineSizeKb = f->ThreadLocalQuarantineSizeKb;
  DeallocationTypeMismatch = f->DeallocationTypeMismatch;
  DeleteSizeMismatch = f->DeleteSizeMismatch;
  ZeroContents = f->ZeroContents;
}

void AllocatorOptions::copyTo(Flags *f, CommonFlags *cf) const {
  cf->allocator_may_return_null = MayReturnNull;
  f->QuarantineSizeMb = QuarantineSizeMb;
  f->ThreadLocalQuarantineSizeKb = ThreadLocalQuarantineSizeKb;
  f->DeallocationTypeMismatch = DeallocationTypeMismatch;
  f->DeleteSizeMismatch = DeleteSizeMismatch;
  f->ZeroContents = ZeroContents;
}

struct Allocator {
  static const uptr MaxAllowedMallocSize = 1ULL << 40;
  static const uptr MinAlignment = 1 << MinAlignmentLog;
  static const uptr MaxAlignment = 1 << MaxAlignmentLog; // 16 MB

  ScudoAllocator BackendAllocator;
  ScudoQuarantine AllocatorQuarantine;

  // The fallback caches are used when the thread local caches have been
  // 'detroyed' on thread tear-down. They are protected by a Mutex as they can
  // be accessed by different threads.
  StaticSpinMutex FallbackMutex;
  AllocatorCache FallbackAllocatorCache;
  QuarantineCache FallbackQuarantineCache;

  bool DeallocationTypeMismatch;
  bool ZeroContents;
  bool DeleteSizeMismatch;

  explicit Allocator(LinkerInitialized)
    : AllocatorQuarantine(LINKER_INITIALIZED),
      FallbackQuarantineCache(LINKER_INITIALIZED) {}

  void init(const AllocatorOptions &Options) {
    // Currently SSE 4.2 support is required. This might change later.
    CHECK(testCPUFeature(SSE4_2)); // for crc32

    // Verify that the header offset field can hold the maximum offset. In the
    // worst case scenario, the backend allocation is already aligned on
    // MaxAlignment, so in order to store the header and still be aligned, we
    // add an extra MaxAlignment. As a result, the offset from the beginning of
    // the backend allocation to the chunk will be MaxAlignment -
    // ChunkHeaderSize.
    UnpackedHeader Header = {};
    uptr MaximumOffset = (MaxAlignment - ChunkHeaderSize) >> MinAlignmentLog;
    Header.Offset = MaximumOffset;
    if (Header.Offset != MaximumOffset) {
      dieWithMessage("ERROR: the maximum possible offset doesn't fit in the "
                     "header\n");
    }

    DeallocationTypeMismatch = Options.DeallocationTypeMismatch;
    DeleteSizeMismatch = Options.DeleteSizeMismatch;
    ZeroContents = Options.ZeroContents;
    BackendAllocator.Init(Options.MayReturnNull);
    AllocatorQuarantine.Init(static_cast<uptr>(Options.QuarantineSizeMb) << 20,
                             static_cast<uptr>(
                                 Options.ThreadLocalQuarantineSizeKb) << 10);
    BackendAllocator.InitCache(&FallbackAllocatorCache);
    Cookie = Prng.Next();
  }

  // Allocates a chunk.
  void *allocate(uptr Size, uptr Alignment, AllocType Type) {
    if (UNLIKELY(!ThreadInited))
      initThread();
    if (!IsPowerOfTwo(Alignment)) {
      dieWithMessage("ERROR: malloc alignment is not a power of 2\n");
    }
    if (Alignment > MaxAlignment)
      return BackendAllocator.ReturnNullOrDie();
    if (Alignment < MinAlignment)
      Alignment = MinAlignment;
    if (Size == 0)
      Size = 1;
    if (Size >= MaxAllowedMallocSize)
      return BackendAllocator.ReturnNullOrDie();
    uptr RoundedSize = RoundUpTo(Size, MinAlignment);
    uptr ExtraBytes = ChunkHeaderSize;
    if (Alignment > MinAlignment)
      ExtraBytes += Alignment;
    uptr NeededSize = RoundedSize + ExtraBytes;
    if (NeededSize >= MaxAllowedMallocSize)
      return BackendAllocator.ReturnNullOrDie();

    void *Ptr;
    if (LIKELY(!ThreadTornDown)) {
      Ptr = BackendAllocator.Allocate(&Cache, NeededSize, MinAlignment);
    } else {
      SpinMutexLock l(&FallbackMutex);
      Ptr = BackendAllocator.Allocate(&FallbackAllocatorCache, NeededSize,
                                      MinAlignment);
    }
    if (!Ptr)
      return BackendAllocator.ReturnNullOrDie();

    // If requested, we will zero out the entire contents of the returned chunk.
    if (ZeroContents && BackendAllocator.FromPrimary(Ptr))
       memset(Ptr, 0, BackendAllocator.GetActuallyAllocatedSize(Ptr));

    uptr AllocBeg = reinterpret_cast<uptr>(Ptr);
    uptr ChunkBeg = AllocBeg + ChunkHeaderSize;
    if (!IsAligned(ChunkBeg, Alignment))
      ChunkBeg = RoundUpTo(ChunkBeg, Alignment);
    CHECK_LE(ChunkBeg + Size, AllocBeg + NeededSize);
    ScudoChunk *Chunk =
        reinterpret_cast<ScudoChunk *>(ChunkBeg - ChunkHeaderSize);
    UnpackedHeader Header = {};
    Header.State = ChunkAllocated;
    Header.Offset = (ChunkBeg - ChunkHeaderSize - AllocBeg) >> MinAlignmentLog;
    Header.AllocType = Type;
    Header.RequestedSize = Size;
    Header.Salt = static_cast<u16>(Prng.Next());
    Chunk->storeHeader(&Header);
    void *UserPtr = reinterpret_cast<void *>(ChunkBeg);
    // TODO(kostyak): hooks sound like a terrible idea security wise but might
    //                be needed for things to work properly?
    // if (&__sanitizer_malloc_hook) __sanitizer_malloc_hook(UserPtr, Size);
    return UserPtr;
  }

  // Deallocates a Chunk, which means adding it to the delayed free list (or
  // Quarantine).
  void deallocate(void *UserPtr, uptr DeleteSize, AllocType Type) {
    if (UNLIKELY(!ThreadInited))
      initThread();
    // TODO(kostyak): see hook comment above
    // if (&__sanitizer_free_hook) __sanitizer_free_hook(UserPtr);
    if (!UserPtr)
      return;
    uptr ChunkBeg = reinterpret_cast<uptr>(UserPtr);
    if (!IsAligned(ChunkBeg, MinAlignment)) {
      dieWithMessage("ERROR: attempted to deallocate a chunk not properly "
                     "aligned at address %p\n", UserPtr);
    }
    ScudoChunk *Chunk =
        reinterpret_cast<ScudoChunk *>(ChunkBeg - ChunkHeaderSize);
    UnpackedHeader OldHeader;
    Chunk->loadHeader(&OldHeader);
    if (OldHeader.State != ChunkAllocated) {
      dieWithMessage("ERROR: invalid chunk state when deallocating address "
                     "%p\n", Chunk);
    }
    UnpackedHeader NewHeader = OldHeader;
    NewHeader.State = ChunkQuarantine;
    Chunk->compareExchangeHeader(&NewHeader, &OldHeader);
    if (DeallocationTypeMismatch) {
      // The deallocation type has to match the allocation one.
      if (NewHeader.AllocType != Type) {
        // With the exception of memalign'd Chunks, that can be still be free'd.
        if (NewHeader.AllocType != FromMemalign || Type != FromMalloc) {
          dieWithMessage("ERROR: allocation type mismatch on address %p\n",
                         Chunk);
        }
      }
    }
    uptr Size = NewHeader.RequestedSize;
    if (DeleteSizeMismatch) {
      if (DeleteSize && DeleteSize != Size) {
        dieWithMessage("ERROR: invalid sized delete on chunk at address %p\n",
                       Chunk);
      }
    }
    if (LIKELY(!ThreadTornDown)) {
      AllocatorQuarantine.Put(&ThreadQuarantineCache,
                              QuarantineCallback(&Cache), Chunk, Size);
    } else {
      SpinMutexLock l(&FallbackMutex);
      AllocatorQuarantine.Put(&FallbackQuarantineCache,
                              QuarantineCallback(&FallbackAllocatorCache),
                              Chunk, Size);
    }
  }

  // Returns the actual usable size of a chunk. Since this requires loading the
  // header, we will return it in the second parameter, as it can be required
  // by the caller to perform additional processing.
  uptr getUsableSize(const void *Ptr, UnpackedHeader *Header) {
    if (UNLIKELY(!ThreadInited))
      initThread();
    if (!Ptr)
      return 0;
    uptr ChunkBeg = reinterpret_cast<uptr>(Ptr);
    ScudoChunk *Chunk =
        reinterpret_cast<ScudoChunk *>(ChunkBeg - ChunkHeaderSize);
    Chunk->loadHeader(Header);
    // Getting the usable size of a chunk only makes sense if it's allocated.
    if (Header->State != ChunkAllocated) {
      dieWithMessage("ERROR: attempted to size a non-allocated chunk at "
                     "address %p\n", Chunk);
    }
    uptr Size =
        BackendAllocator.GetActuallyAllocatedSize(Chunk->AllocBeg(Header));
    // UsableSize works as malloc_usable_size, which is also what (AFAIU)
    // tcmalloc's MallocExtension::GetAllocatedSize aims at providing. This
    // means we will return the size of the chunk from the user beginning to
    // the end of the 'user' allocation, hence us subtracting the header size
    // and the offset from the size.
    if (Size == 0)
      return Size;
    return Size - ChunkHeaderSize - (Header->Offset << MinAlignmentLog);
  }

  // Helper function that doesn't care about the header.
  uptr getUsableSize(const void *Ptr) {
    UnpackedHeader Header;
    return getUsableSize(Ptr, &Header);
  }

  // Reallocates a chunk. We can save on a new allocation if the new requested
  // size still fits in the chunk.
  void *reallocate(void *OldPtr, uptr NewSize) {
    if (UNLIKELY(!ThreadInited))
      initThread();
    UnpackedHeader OldHeader;
    uptr Size = getUsableSize(OldPtr, &OldHeader);
    uptr ChunkBeg = reinterpret_cast<uptr>(OldPtr);
    ScudoChunk *Chunk =
        reinterpret_cast<ScudoChunk *>(ChunkBeg - ChunkHeaderSize);
    if (OldHeader.AllocType != FromMalloc) {
      dieWithMessage("ERROR: invalid chunk type when reallocating address %p\n",
                     Chunk);
    }
    UnpackedHeader NewHeader = OldHeader;
    // The new size still fits in the current chunk.
    if (NewSize <= Size) {
      NewHeader.RequestedSize = NewSize;
      Chunk->compareExchangeHeader(&NewHeader, &OldHeader);
      return OldPtr;
    }
    // Otherwise, we have to allocate a new chunk and copy the contents of the
    // old one.
    void *NewPtr = allocate(NewSize, MinAlignment, FromMalloc);
    if (NewPtr) {
      uptr OldSize = OldHeader.RequestedSize;
      memcpy(NewPtr, OldPtr, Min(NewSize, OldSize));
      NewHeader.State = ChunkQuarantine;
      Chunk->compareExchangeHeader(&NewHeader, &OldHeader);
      if (LIKELY(!ThreadTornDown)) {
        AllocatorQuarantine.Put(&ThreadQuarantineCache,
                                QuarantineCallback(&Cache), Chunk, OldSize);
      } else {
        SpinMutexLock l(&FallbackMutex);
        AllocatorQuarantine.Put(&FallbackQuarantineCache,
                                QuarantineCallback(&FallbackAllocatorCache),
                                Chunk, OldSize);
      }
    }
    return NewPtr;
  }

  void *calloc(uptr NMemB, uptr Size) {
    if (UNLIKELY(!ThreadInited))
      initThread();
    uptr Total = NMemB * Size;
    if (Size != 0 && Total / Size != NMemB) // Overflow check
      return BackendAllocator.ReturnNullOrDie();
    void *Ptr = allocate(Total, MinAlignment, FromMalloc);
    // If ZeroContents, the content of the chunk has already been zero'd out.
    if (!ZeroContents && Ptr && BackendAllocator.FromPrimary(Ptr))
      memset(Ptr, 0, getUsableSize(Ptr));
    return Ptr;
  }

  void drainQuarantine() {
    AllocatorQuarantine.Drain(&ThreadQuarantineCache,
                              QuarantineCallback(&Cache));
  }
};

static Allocator Instance(LINKER_INITIALIZED);

static ScudoAllocator &getAllocator() {
  return Instance.BackendAllocator;
}

void initAllocator(const AllocatorOptions &Options) {
  Instance.init(Options);
}

void drainQuarantine() {
  Instance.drainQuarantine();
}

void *scudoMalloc(uptr Size, AllocType Type) {
  return Instance.allocate(Size, Allocator::MinAlignment, Type);
}

void scudoFree(void *Ptr, AllocType Type) {
  Instance.deallocate(Ptr, 0, Type);
}

void scudoSizedFree(void *Ptr, uptr Size, AllocType Type) {
  Instance.deallocate(Ptr, Size, Type);
}

void *scudoRealloc(void *Ptr, uptr Size) {
  if (!Ptr)
    return Instance.allocate(Size, Allocator::MinAlignment, FromMalloc);
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

} // namespace __scudo

using namespace __scudo;

// MallocExtension helper functions

uptr __sanitizer_get_current_allocated_bytes() {
  uptr stats[AllocatorStatCount];
  getAllocator().GetStats(stats);
  return stats[AllocatorStatAllocated];
}

uptr __sanitizer_get_heap_size() {
  uptr stats[AllocatorStatCount];
  getAllocator().GetStats(stats);
  return stats[AllocatorStatMapped];
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

int __sanitizer_get_ownership(const void *p) {
  return Instance.getUsableSize(p) != 0;
}

uptr __sanitizer_get_allocated_size(const void *p) {
  return Instance.getUsableSize(p);
}
