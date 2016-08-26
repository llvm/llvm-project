//===-- asan_allocator.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// ASan-private header for asan_allocator.cc.
//===----------------------------------------------------------------------===//

#ifndef ASAN_ALLOCATOR_H
#define ASAN_ALLOCATOR_H

#include "asan_flags.h"
#include "asan_internal.h"
#include "asan_interceptors.h"
#include "sanitizer_common/sanitizer_allocator.h"
#include "sanitizer_common/sanitizer_list.h"

namespace __asan {

enum AllocType {
  FROM_MALLOC = 1,  // Memory block came from malloc, calloc, realloc, etc.
  FROM_NEW = 2,     // Memory block came from operator new.
  FROM_NEW_BR = 3   // Memory block came from operator new [ ]
};

struct AsanChunk;

struct AllocatorOptions {
  u32 quarantine_size_mb;
  u16 min_redzone;
  u16 max_redzone;
  u8 may_return_null;
  u8 alloc_dealloc_mismatch;

  void SetFrom(const Flags *f, const CommonFlags *cf);
  void CopyTo(Flags *f, CommonFlags *cf);
};

void InitializeAllocator(const AllocatorOptions &options);
void ReInitializeAllocator(const AllocatorOptions &options);
void GetAllocatorOptions(AllocatorOptions *options);

class AsanChunkView {
 public:
  explicit AsanChunkView(AsanChunk *chunk) : chunk_(chunk) {}
  bool IsValid();        // Checks if AsanChunkView points to a valid allocated
                         // or quarantined chunk.
  bool IsAllocated();    // Checks if the memory is currently allocated.
  uptr Beg();            // First byte of user memory.
  uptr End();            // Last byte of user memory.
  uptr UsedSize();       // Size requested by the user.
  uptr AllocTid();
  uptr FreeTid();
  bool Eq(const AsanChunkView &c) const { return chunk_ == c.chunk_; }
  u32 GetAllocStackId();
  u32 GetFreeStackId();
  StackTrace GetAllocStack();
  StackTrace GetFreeStack();
  AllocType GetAllocType();
  bool AddrIsInside(uptr addr, uptr access_size, sptr *offset) {
    if (addr >= Beg() && (addr + access_size) <= End()) {
      *offset = addr - Beg();
      return true;
    }
    return false;
  }
  bool AddrIsAtLeft(uptr addr, uptr access_size, sptr *offset) {
    (void)access_size;
    if (addr < Beg()) {
      *offset = Beg() - addr;
      return true;
    }
    return false;
  }
  bool AddrIsAtRight(uptr addr, uptr access_size, sptr *offset) {
    if (addr + access_size > End()) {
      *offset = addr - End();
      return true;
    }
    return false;
  }

 private:
  AsanChunk *const chunk_;
};

AsanChunkView FindHeapChunkByAddress(uptr address);
AsanChunkView FindHeapChunkByAllocBeg(uptr address);

// List of AsanChunks with total size.
class AsanChunkFifoList: public IntrusiveList<AsanChunk> {
 public:
  explicit AsanChunkFifoList(LinkerInitialized) { }
  AsanChunkFifoList() { clear(); }
  void Push(AsanChunk *n);
  void PushList(AsanChunkFifoList *q);
  AsanChunk *Pop();
  uptr size() { return size_; }
  void clear() {
    IntrusiveList<AsanChunk>::clear();
    size_ = 0;
  }
 private:
  uptr size_;
};

struct AsanMapUnmapCallback {
  void OnMap(uptr p, uptr size) const;
  void OnUnmap(uptr p, uptr size) const;
};

#if SANITIZER_CAN_USE_ALLOCATOR64
# if defined(__powerpc64__)
const uptr kAllocatorSpace =  0xa0000000000ULL;
const uptr kAllocatorSize  =  0x20000000000ULL;  // 2T.
# elif defined(__aarch64__)
// AArch64/SANITIZIER_CAN_USER_ALLOCATOR64 is only for 42-bit VMA
// so no need to different values for different VMA.
const uptr kAllocatorSpace =  0x10000000000ULL;
const uptr kAllocatorSize  =  0x10000000000ULL;  // 3T.
# else
const uptr kAllocatorSpace = 0x600000000000ULL;
const uptr kAllocatorSize  =  0x40000000000ULL;  // 4T.
# endif
typedef DefaultSizeClassMap SizeClassMap;
struct AP64 {  // Allocator64 parameters. Deliberately using a short name.
  static const uptr kSpaceBeg = kAllocatorSpace;
  static const uptr kSpaceSize = kAllocatorSize;
  static const uptr kMetadataSize = 0;
  typedef __asan::SizeClassMap SizeClassMap;
  typedef AsanMapUnmapCallback MapUnmapCallback;
  static const uptr kFlags = 0;
};

typedef SizeClassAllocator64<AP64> PrimaryAllocator;
#else  // Fallback to SizeClassAllocator32.
static const uptr kRegionSizeLog = 20;
static const uptr kNumRegions = SANITIZER_MMAP_RANGE_SIZE >> kRegionSizeLog;
# if SANITIZER_WORDSIZE == 32
typedef FlatByteMap<kNumRegions> ByteMap;
# elif SANITIZER_WORDSIZE == 64
typedef TwoLevelByteMap<(kNumRegions >> 12), 1 << 12> ByteMap;
# endif
typedef CompactSizeClassMap SizeClassMap;
typedef SizeClassAllocator32<0, SANITIZER_MMAP_RANGE_SIZE, 16,
  SizeClassMap, kRegionSizeLog,
  ByteMap,
  AsanMapUnmapCallback> PrimaryAllocator;
#endif  // SANITIZER_CAN_USE_ALLOCATOR64

static const uptr kNumberOfSizeClasses = SizeClassMap::kNumClasses;
typedef SizeClassAllocatorLocalCache<PrimaryAllocator> AllocatorCache;
typedef LargeMmapAllocator<AsanMapUnmapCallback> SecondaryAllocator;
typedef CombinedAllocator<PrimaryAllocator, AllocatorCache,
    SecondaryAllocator> AsanAllocator;


struct AsanThreadLocalMallocStorage {
  uptr quarantine_cache[16];
  AllocatorCache allocator_cache;
  void CommitBack();
 private:
  // These objects are allocated via mmap() and are zero-initialized.
  AsanThreadLocalMallocStorage() {}
};

void *asan_memalign(uptr alignment, uptr size, BufferedStackTrace *stack,
                    AllocType alloc_type);
void asan_free(void *ptr, BufferedStackTrace *stack, AllocType alloc_type);
void asan_sized_free(void *ptr, uptr size, BufferedStackTrace *stack,
                     AllocType alloc_type);

void *asan_malloc(uptr size, BufferedStackTrace *stack);
void *asan_calloc(uptr nmemb, uptr size, BufferedStackTrace *stack);
void *asan_realloc(void *p, uptr size, BufferedStackTrace *stack);
void *asan_valloc(uptr size, BufferedStackTrace *stack);
void *asan_pvalloc(uptr size, BufferedStackTrace *stack);

int asan_posix_memalign(void **memptr, uptr alignment, uptr size,
                        BufferedStackTrace *stack);
uptr asan_malloc_usable_size(const void *ptr, uptr pc, uptr bp);

uptr asan_mz_size(const void *ptr);
void asan_mz_force_lock();
void asan_mz_force_unlock();

void PrintInternalAllocatorStats();
void AsanSoftRssLimitExceededCallback(bool exceeded);

}  // namespace __asan
#endif  // ASAN_ALLOCATOR_H
