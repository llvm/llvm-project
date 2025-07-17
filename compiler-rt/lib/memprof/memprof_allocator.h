//===-- memprof_allocator.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of MemProfiler, a memory profiler.
//
// MemProf-private header for memprof_allocator.cpp.
//===----------------------------------------------------------------------===//

#ifndef MEMPROF_ALLOCATOR_H
#define MEMPROF_ALLOCATOR_H

#include "memprof_flags.h"
#include "memprof_interceptors.h"
#include "memprof_internal.h"
#include "sanitizer_common/sanitizer_allocator.h"
#include "sanitizer_common/sanitizer_list.h"

namespace __memprof {

enum AllocType {
  FROM_MALLOC = 1, // Memory block came from malloc, calloc, realloc, etc.
  FROM_NEW = 2,    // Memory block came from operator new.
  FROM_NEW_BR = 3  // Memory block came from operator new [ ]
};

void InitializeAllocator();

struct MemprofMapUnmapCallback {
  void OnMap(uptr p, uptr size) const;
  void OnMapSecondary(uptr p, uptr size, uptr user_begin,
                      uptr user_size) const {
    OnMap(p, size);
  }
  void OnUnmap(uptr p, uptr size) const;
};

#if SANITIZER_CAN_USE_ALLOCATOR64
constexpr uptr kAllocatorSpace = ~(uptr)0;
constexpr uptr kAllocatorSize = 0x40000000000ULL; // 4T.
typedef DefaultSizeClassMap SizeClassMap;
template <typename AddressSpaceViewTy>
struct AP64 { // Allocator64 parameters. Deliberately using a short name.
  static const uptr kSpaceBeg = kAllocatorSpace;
  static const uptr kSpaceSize = kAllocatorSize;
  static const uptr kMetadataSize = 0;
  typedef __memprof::SizeClassMap SizeClassMap;
  typedef MemprofMapUnmapCallback MapUnmapCallback;
  static const uptr kFlags = 0;
  using AddressSpaceView = AddressSpaceViewTy;
};

template <typename AddressSpaceView>
using PrimaryAllocatorASVT = SizeClassAllocator64<AP64<AddressSpaceView>>;
using PrimaryAllocator = PrimaryAllocatorASVT<LocalAddressSpaceView>;

#else // Fallback to SizeClassAllocator32.
typedef CompactSizeClassMap SizeClassMap;
template <typename AddressSpaceViewTy> struct AP32 {
  static const uptr kSpaceBeg = 0;
  static const u64 kSpaceSize = SANITIZER_MMAP_RANGE_SIZE;
  static const uptr kMetadataSize = 0;
  typedef __memprof::SizeClassMap SizeClassMap;
  static const uptr kRegionSizeLog = 20;
  using AddressSpaceView = AddressSpaceViewTy;
  typedef MemprofMapUnmapCallback MapUnmapCallback;
  static const uptr kFlags = 0;
};
template <typename AddressSpaceView>
using PrimaryAllocatorASVT = SizeClassAllocator32<AP32<AddressSpaceView>>;
using PrimaryAllocator = PrimaryAllocatorASVT<LocalAddressSpaceView>;
#endif

static const uptr kNumberOfSizeClasses = SizeClassMap::kNumClasses;

template <typename AddressSpaceView>
using MemprofAllocatorASVT =
    CombinedAllocator<PrimaryAllocatorASVT<AddressSpaceView>>;
using MemprofAllocator = MemprofAllocatorASVT<LocalAddressSpaceView>;
using AllocatorCache = MemprofAllocator::AllocatorCache;

struct MemprofThreadLocalMallocStorage {
  AllocatorCache allocator_cache;
  void CommitBack();

private:
  // These objects are allocated via mmap() and are zero-initialized.
  MemprofThreadLocalMallocStorage() {}
};

void *memprof_memalign(uptr alignment, uptr size, BufferedStackTrace *stack,
                       AllocType alloc_type);
void memprof_free(void *ptr, BufferedStackTrace *stack, AllocType alloc_type);
void memprof_delete(void *ptr, uptr size, uptr alignment,
                    BufferedStackTrace *stack, AllocType alloc_type);

void *memprof_malloc(uptr size, BufferedStackTrace *stack);
void *memprof_calloc(uptr nmemb, uptr size, BufferedStackTrace *stack);
void *memprof_realloc(void *p, uptr size, BufferedStackTrace *stack);
void *memprof_reallocarray(void *p, uptr nmemb, uptr size,
                           BufferedStackTrace *stack);
void *memprof_valloc(uptr size, BufferedStackTrace *stack);
void *memprof_pvalloc(uptr size, BufferedStackTrace *stack);

void *memprof_aligned_alloc(uptr alignment, uptr size,
                            BufferedStackTrace *stack);
int memprof_posix_memalign(void **memptr, uptr alignment, uptr size,
                           BufferedStackTrace *stack);
uptr memprof_malloc_usable_size(const void *ptr);

uptr memprof_mz_size(const void *ptr);
void memprof_mz_force_lock();
void memprof_mz_force_unlock();

void PrintInternalAllocatorStats();

} // namespace __memprof
#endif // MEMPROF_ALLOCATOR_H
