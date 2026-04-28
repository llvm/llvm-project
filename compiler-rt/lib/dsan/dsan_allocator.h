//=-- dsan_allocator.h ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of DoubleFreeSanitizer.
// Allocator for standalone DSan.
//
//===----------------------------------------------------------------------===//

#ifndef DSAN_ALLOCATOR_H
#define DSAN_ALLOCATOR_H

#include "sanitizer_common/sanitizer_allocator.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_internal_defs.h"

namespace __dsan {

void* Allocate(const StackTrace& stack, uptr size, uptr alignment,
               bool cleared);
void Deallocate(const StackTrace& stack, void* p);
void* Reallocate(const StackTrace& stack, void* p, uptr new_size,
                 uptr alignment);
uptr GetMallocUsableSize(const void* p);

void GetAllocatorCacheRange(uptr* begin, uptr* end);
void AllocatorThreadStart();
void AllocatorThreadFinish();
void InitializeAllocator();

const bool kAlwaysClearMemory = true;

struct ChunkMetadata {
  u8 state;  // Must be first. See ChunkState.
#if SANITIZER_WORDSIZE == 64
  uptr requested_size : 56;
#else
  uptr requested_size : 32;
  uptr padding2 : 24;
#endif
  u32 alloc_stack_id;  // Stack trace of the allocation
  u32 free_stack_id;   // Stack trace of the first free
};

enum ChunkState : u8 {
  kChunkInvalid,
  kChunkAllocated,
  kChunkFreeing,
  kChunkFreed,
};

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
#  if SANITIZER_FUCHSIA || defined(__powerpc64__)
const uptr kAllocatorSpace = ~(uptr)0;
#    if SANITIZER_RISCV64
// See the comments in compiler-rt/lib/asan/asan_allocator.h for why these
// values were chosen.
const uptr kAllocatorSize = UINT64_C(1) << 33;  // 8GB
using DSanSizeClassMap = SizeClassMap</*kNumBits=*/2,
                                      /*kMinSizeLog=*/5,
                                      /*kMidSizeLog=*/8,
                                      /*kMaxSizeLog=*/18,
                                      /*kNumCachedHintT=*/8,
                                      /*kMaxBytesCachedLog=*/10>;
static_assert(DSanSizeClassMap::kNumClassesRounded <= 32,
              "32 size classes is the optimal number to ensure tests run "
              "efficiently on Fuchsia.");
#    else
const uptr kAllocatorSize = 0x40000000000ULL;  // 4T.
using DSanSizeClassMap = DefaultSizeClassMap;
#    endif
#  elif SANITIZER_RISCV64
const uptr kAllocatorSpace = ~(uptr)0;
const uptr kAllocatorSize = 0x2000000000ULL;  // 128G.
using DSanSizeClassMap = DefaultSizeClassMap;
#  elif SANITIZER_APPLE
const uptr kAllocatorSpace = 0x600000000000ULL;
const uptr kAllocatorSize = 0x40000000000ULL;  // 4T.
using DSanSizeClassMap = DefaultSizeClassMap;
#  elif SANITIZER_ANDROID && defined(__aarch64__)
const uptr kAllocatorSpace = 0x3000000000ULL;
const uptr kAllocatorSize = 0x2000000000ULL;
using DSanSizeClassMap = VeryCompactSizeClassMap;
#  else
const uptr kAllocatorSpace = 0x500000000000ULL;
const uptr kAllocatorSize = 0x40000000000ULL;  // 4T.
using DSanSizeClassMap = DefaultSizeClassMap;
#  endif
template <typename AddressSpaceViewTy>
struct AP64 {  // Allocator64 parameters. Deliberately using a short name.
  static const uptr kSpaceBeg = kAllocatorSpace;
  static const uptr kSpaceSize = kAllocatorSize;
  static const uptr kMetadataSize = sizeof(ChunkMetadata);
  using SizeClassMap = DSanSizeClassMap;
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

Allocator::AllocatorCache* GetAllocatorCache();

int dsan_posix_memalign(void** memptr, uptr alignment, uptr size,
                        const StackTrace& stack);
void* dsan_aligned_alloc(uptr alignment, uptr size, const StackTrace& stack);
void* dsan_memalign(uptr alignment, uptr size, const StackTrace& stack);
void* dsan_malloc(uptr size, const StackTrace& stack);
void dsan_free(void* p, const StackTrace& stack);
void dsan_free_sized(void* p, uptr size, const StackTrace& stack);
void dsan_free_aligned_sized(void* p, uptr alignment, uptr size,
                             const StackTrace& stack);
void* dsan_realloc(void* p, uptr size, const StackTrace& stack);
void* dsan_reallocarray(void* p, uptr nmemb, uptr size,
                        const StackTrace& stack);
void* dsan_calloc(uptr nmemb, uptr size, const StackTrace& stack);
void* dsan_valloc(uptr size, const StackTrace& stack);
void* dsan_pvalloc(uptr size, const StackTrace& stack);
uptr dsan_mz_size(const void* p);

}  // namespace __dsan

#endif  // DSAN_ALLOCATOR_H
