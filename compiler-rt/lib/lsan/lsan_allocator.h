//=-- lsan_allocator.h ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of LeakSanitizer.
// Allocator for standalone LSan.
//
//===----------------------------------------------------------------------===//

#ifndef LSAN_ALLOCATOR_H
#define LSAN_ALLOCATOR_H

#include "sanitizer_common/sanitizer_allocator.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "lsan_common.h"

namespace __lsan {

void *Allocate(const StackTrace &stack, uptr size, uptr alignment,
               bool cleared);
void Deallocate(void *p, const StackTrace *free_stack);
void *Reallocate(const StackTrace &stack, void *p, uptr new_size,
                 uptr alignment);
uptr GetMallocUsableSize(const void *p);

template<typename Callable>
void ForEachChunk(const Callable &callback);

void GetAllocatorCacheRange(uptr *begin, uptr *end);
void AllocatorThreadStart();
void AllocatorThreadFinish();
void InitializeAllocator();

// Locks protecting the double-free side table, for the fork handlers.
void LockDoubleFree();
void UnlockDoubleFree();

const bool kAlwaysClearMemory = true;

// Lifetime of a chunk, stored in ChunkMetadata::allocated.
//
// kChunkFreeing is a transient state used by double-free detection: the thread
// that won the race to free the chunk has claimed it but has not published
// ChunkMetadata::free_stack_id yet.
enum ChunkState : u8 {
  kChunkFree = 0,
  kChunkAllocated = 1,
  kChunkFreeing = 2,
};

// Appending free_stack_id to the natural 32-bit layout would have grown the
// per-chunk metadata to 20 bytes, so requested_size is stored as a separate
// word there and the leftover bitfield space is left unused.
//
// `allocated` is accessed atomically through the first byte of the struct, so
// it shares its storage unit with the bitfields that follow it and a plain
// store to one of those is a read-modify-write over it. Every such store made
// while a chunk can be freed concurrently would therefore have to go through
// the same atomic word. Today only IgnoreObject() does that (see the comment
// there); the allocation path writes the bitfields while the chunk is still
// kChunkFree, where a losing compare-exchange writes nothing, and the free
// path never touches them.
struct ChunkMetadata {
  u8 allocated : 8;  // Must be first. Holds a ChunkState.
  ChunkTag tag : 2;
#if SANITIZER_WORDSIZE == 64
  uptr requested_size : 54;
#else
  uptr padding : 22;
  u32 requested_size;
#endif
  u32 stack_trace_id;
  // Stack of the first free().
  u32 free_stack_id;
};

static_assert(sizeof(ChunkMetadata) == 4 * sizeof(u32),
              "ChunkMetadata must stay 16 bytes: it is stored for every "
              "allocator chunk.");

#if !SANITIZER_CAN_USE_ALLOCATOR64
template <typename AddressSpaceViewTy>
struct AP32 {
  static const uptr kSpaceBeg = SANITIZER_MMAP_BEGIN;
  static const u64 kSpaceSize = SANITIZER_MMAP_RANGE_SIZE;
  static const uptr kMetadataSize = sizeof(ChunkMetadata);
  typedef __sanitizer::CompactSizeClassMap SizeClassMap;
  static const uptr kRegionSizeLog = 20;
  using AddressSpaceView = AddressSpaceViewTy;
  typedef NoOpMapUnmapCallback MapUnmapCallback;
  static const uptr kFlags = 0;
};
template <typename AddressSpaceView>
using PrimaryAllocatorASVT = SizeClassAllocator32<AP32<AddressSpaceView>>;
using PrimaryAllocator = PrimaryAllocatorASVT<LocalAddressSpaceView>;
#else
# if SANITIZER_FUCHSIA || defined(__powerpc64__)
const uptr kAllocatorSpace = ~(uptr)0;
#    if SANITIZER_RISCV64
// See the comments in compiler-rt/lib/asan/asan_allocator.h for why these
// values were chosen.
const uptr kAllocatorSize = UINT64_C(1) << 33;  // 8GB
using LSanSizeClassMap = SizeClassMap</*kNumBits=*/2,
                                      /*kMinSizeLog=*/5,
                                      /*kMidSizeLog=*/8,
                                      /*kMaxSizeLog=*/18,
                                      /*kNumCachedHintT=*/8,
                                      /*kMaxBytesCachedLog=*/10>;
static_assert(LSanSizeClassMap::kNumClassesRounded <= 32,
              "32 size classes is the optimal number to ensure tests run "
              "effieciently on Fuchsia.");
#    else
const uptr kAllocatorSize  =  0x40000000000ULL;  // 4T.
using LSanSizeClassMap = DefaultSizeClassMap;
#    endif
#  elif SANITIZER_RISCV64
const uptr kAllocatorSpace = ~(uptr)0;
const uptr kAllocatorSize = 0x2000000000ULL;  // 128G.
using LSanSizeClassMap = DefaultSizeClassMap;
#  elif SANITIZER_APPLE
const uptr kAllocatorSpace = 0x600000000000ULL;
const uptr kAllocatorSize  = 0x40000000000ULL;  // 4T.
using LSanSizeClassMap = DefaultSizeClassMap;
#  elif SANITIZER_ANDROID && defined(__aarch64__)
const uptr kAllocatorSpace = 0x3000000000ULL;
const uptr kAllocatorSize = 0x2000000000ULL;
using LSanSizeClassMap = VeryCompactSizeClassMap;
#  else
const uptr kAllocatorSpace = 0x500000000000ULL;
const uptr kAllocatorSize = 0x40000000000ULL;  // 4T.
using LSanSizeClassMap = DefaultSizeClassMap;
#  endif
template <typename AddressSpaceViewTy>
struct AP64 {  // Allocator64 parameters. Deliberately using a short name.
  static const uptr kSpaceBeg = kAllocatorSpace;
  static const uptr kSpaceSize = kAllocatorSize;
  static const uptr kMetadataSize = sizeof(ChunkMetadata);
  using SizeClassMap = LSanSizeClassMap;
  typedef NoOpMapUnmapCallback MapUnmapCallback;
  static const uptr kFlags = 0;
  using AddressSpaceView = AddressSpaceViewTy;
};

template <typename AddressSpaceView>
using PrimaryAllocatorASVT = SizeClassAllocator64<AP64<AddressSpaceView>>;
using PrimaryAllocator = PrimaryAllocatorASVT<LocalAddressSpaceView>;
#endif

template <typename AddressSpaceView>
using AllocatorASVT = CombinedAllocator<PrimaryAllocatorASVT<AddressSpaceView>>;
using Allocator = AllocatorASVT<LocalAddressSpaceView>;
using AllocatorCache = Allocator::AllocatorCache;

Allocator::AllocatorCache *GetAllocatorCache();

int lsan_posix_memalign(void **memptr, uptr alignment, uptr size,
                        const StackTrace &stack);
void *lsan_aligned_alloc(uptr alignment, uptr size, const StackTrace &stack);
void *lsan_memalign(uptr alignment, uptr size, const StackTrace &stack);
void *lsan_malloc(uptr size, const StackTrace &stack);
void lsan_free(void *p, const StackTrace *free_stack);
void lsan_free_sized(void *p, uptr size, const StackTrace *free_stack);
void lsan_free_aligned_sized(void *p, uptr alignment, uptr size,
                             const StackTrace *free_stack);

// free() variants that capture the calling stack for double-free reporting.
//
// These are deliberately out of line. A BufferedStackTrace is about 2 KiB, and
// in most of these interceptors the compiler reserves it in the frame on entry
// even though it is only used on one branch. Capturing inline therefore grows
// the frame of most free() and operator delete() calls, including when
// detect_double_free is disabled.
//
// The caller passes its own pc and bp so that the captured stack starts at the
// intercepted function rather than at the helper.
void NOINLINE lsan_free_with_stack(void *p, uptr pc, uptr bp);
void NOINLINE lsan_free_sized_with_stack(void *p, uptr size, uptr pc, uptr bp);
void NOINLINE lsan_free_aligned_sized_with_stack(void *p, uptr alignment,
                                                 uptr size, uptr pc, uptr bp);

void *lsan_realloc(void *p, uptr size, const StackTrace &stack);
void *lsan_reallocarray(void *p, uptr nmemb, uptr size,
                        const StackTrace &stack);
void *lsan_calloc(uptr nmemb, uptr size, const StackTrace &stack);
void *lsan_valloc(uptr size, const StackTrace &stack);
void *lsan_pvalloc(uptr size, const StackTrace &stack);
uptr lsan_mz_size(const void *p);

}  // namespace __lsan

#endif  // LSAN_ALLOCATOR_H
