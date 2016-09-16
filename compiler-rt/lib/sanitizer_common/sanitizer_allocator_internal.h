//===-- sanitizer_allocator_internal.h --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This allocator is used inside run-times.
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_ALLOCATOR_INTERNAL_H
#define SANITIZER_ALLOCATOR_INTERNAL_H

#include "sanitizer_allocator.h"
#include "sanitizer_internal_defs.h"

namespace __sanitizer {

// FIXME: Check if we may use even more compact size class map for internal
// purposes.
typedef CompactSizeClassMap InternalSizeClassMap;

static const uptr kInternalAllocatorSpace = 0;
static const u64 kInternalAllocatorSize = SANITIZER_MMAP_RANGE_SIZE;
static const uptr kInternalAllocatorRegionSizeLog = 20;
#if SANITIZER_WORDSIZE == 32
static const uptr kInternalAllocatorNumRegions =
    kInternalAllocatorSize >> kInternalAllocatorRegionSizeLog;
typedef FlatByteMap<kInternalAllocatorNumRegions> ByteMap;
#else
static const uptr kInternalAllocatorNumRegions =
    kInternalAllocatorSize >> kInternalAllocatorRegionSizeLog;
typedef TwoLevelByteMap<(kInternalAllocatorNumRegions >> 12), 1 << 12> ByteMap;
#endif
typedef SizeClassAllocator32<
    kInternalAllocatorSpace, kInternalAllocatorSize, 0, InternalSizeClassMap,
    kInternalAllocatorRegionSizeLog, ByteMap> PrimaryInternalAllocator;

typedef SizeClassAllocatorLocalCache<PrimaryInternalAllocator>
    InternalAllocatorCache;

typedef CombinedAllocator<PrimaryInternalAllocator, InternalAllocatorCache,
                          LargeMmapAllocator<> > InternalAllocator;

void *InternalAlloc(uptr size, InternalAllocatorCache *cache = nullptr,
                    uptr alignment = 0);
void *InternalRealloc(void *p, uptr size,
                      InternalAllocatorCache *cache = nullptr);
void *InternalCalloc(uptr countr, uptr size,
                     InternalAllocatorCache *cache = nullptr);
void InternalFree(void *p, InternalAllocatorCache *cache = nullptr);
InternalAllocator *internal_allocator();

enum InternalAllocEnum {
  INTERNAL_ALLOC
};

} // namespace __sanitizer

inline void *operator new(__sanitizer::operator_new_size_type size,
                          __sanitizer::InternalAllocEnum) {
  return __sanitizer::InternalAlloc(size);
}

#endif // SANITIZER_ALLOCATOR_INTERNAL_H
