//===-- asan_mapping.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// Defines ASan memory mapping.
//===----------------------------------------------------------------------===//
#ifndef ASAN_MAPPING_H
#define ASAN_MAPPING_H

#include "asan_internal.h"
#include "asan_shadow_defines.h"

#if defined(ASAN_SHADOW_SCALE)
static const u64 kDefaultShadowScale = ASAN_SHADOW_SCALE;
#else
static const u64 kDefaultShadowScale = 3;
#endif

static const u64 kDefaultShadowSentinel = ~(uptr)0;

#if defined(SHADOW_OFFSET_CONST)
static const u64 kConstShadowOffset = SHADOW_OFFSET_CONST;
#  define SHADOW_OFFSET kConstShadowOffset
#else
#  define SHADOW_OFFSET __asan_shadow_memory_dynamic_address
#endif

#define SHADOW_SCALE kDefaultShadowScale

#if SANITIZER_ANDROID && defined(__arm__)
# define ASAN_PREMAP_SHADOW 1
#else
# define ASAN_PREMAP_SHADOW 0
#endif

#define SHADOW_GRANULARITY (1ULL << SHADOW_SCALE)

#define DO_ASAN_MAPPING_PROFILE 0  // Set to 1 to profile the functions below.

#if DO_ASAN_MAPPING_PROFILE
# define PROFILE_ASAN_MAPPING() AsanMappingProfile[__LINE__]++;
#else
# define PROFILE_ASAN_MAPPING()
#endif

// If 1, all shadow boundaries are constants.
// Don't set to 1 other than for testing.
#define ASAN_FIXED_MAPPING 0

namespace __asan {

extern uptr AsanMappingProfile[];

#if ASAN_FIXED_MAPPING
// Fixed mapping for 64-bit Linux. Mostly used for performance comparison
// with non-fixed mapping. As of r175253 (Feb 2013) the performance
// difference between fixed and non-fixed mapping is below the noise level.
static uptr kHighMemEnd = 0x7fffffffffffULL;
static uptr kMidMemBeg =    0x3000000000ULL;
static uptr kMidMemEnd =    0x4fffffffffULL;
#else
extern uptr kHighMemEnd, kMidMemBeg, kMidMemEnd;  // Initialized in __asan_init.
#endif

}  // namespace __asan

#if defined(__sparc__) && SANITIZER_WORDSIZE == 64
#  include "asan_mapping_sparc64.h"
#else
#define MEM_TO_SHADOW(mem) (((mem) >> SHADOW_SCALE) + (SHADOW_OFFSET))

#define kLowMemBeg      0
#define kLowMemEnd      (SHADOW_OFFSET ? SHADOW_OFFSET - 1 : 0)

#define kLowShadowBeg   SHADOW_OFFSET
#define kLowShadowEnd   MEM_TO_SHADOW(kLowMemEnd)

#define kHighMemBeg     (MEM_TO_SHADOW(kHighMemEnd) + 1)

#define kHighShadowBeg  MEM_TO_SHADOW(kHighMemBeg)
#define kHighShadowEnd  MEM_TO_SHADOW(kHighMemEnd)

# define kMidShadowBeg MEM_TO_SHADOW(kMidMemBeg)
# define kMidShadowEnd MEM_TO_SHADOW(kMidMemEnd)

// With the zero shadow base we can not actually map pages starting from 0.
// This constant is somewhat arbitrary.
#define kZeroBaseShadowStart 0
#define kZeroBaseMaxShadowStart (1 << 18)

#define kShadowGapBeg   (kLowShadowEnd ? kLowShadowEnd + 1 \
                                       : kZeroBaseShadowStart)
#define kShadowGapEnd   ((kMidMemBeg ? kMidShadowBeg : kHighShadowBeg) - 1)

#define kShadowGap2Beg (kMidMemBeg ? kMidShadowEnd + 1 : 0)
#define kShadowGap2End (kMidMemBeg ? kMidMemBeg - 1 : 0)

#define kShadowGap3Beg (kMidMemBeg ? kMidMemEnd + 1 : 0)
#define kShadowGap3End (kMidMemBeg ? kHighShadowBeg - 1 : 0)

namespace __asan {

static inline bool AddrIsInLowMem(uptr a) {
  PROFILE_ASAN_MAPPING();
  return a <= kLowMemEnd;
}

static inline bool AddrIsInLowShadow(uptr a) {
  PROFILE_ASAN_MAPPING();
  return a >= kLowShadowBeg && a <= kLowShadowEnd;
}

static inline bool AddrIsInMidMem(uptr a) {
  PROFILE_ASAN_MAPPING();
  return kMidMemBeg && a >= kMidMemBeg && a <= kMidMemEnd;
}

static inline bool AddrIsInMidShadow(uptr a) {
  PROFILE_ASAN_MAPPING();
  return kMidMemBeg && a >= kMidShadowBeg && a <= kMidShadowEnd;
}

static inline bool AddrIsInHighMem(uptr a) {
  PROFILE_ASAN_MAPPING();
  return kHighMemBeg && a >= kHighMemBeg && a <= kHighMemEnd;
}

static inline bool AddrIsInHighShadow(uptr a) {
  PROFILE_ASAN_MAPPING();
  return kHighMemBeg && a >= kHighShadowBeg && a <= kHighShadowEnd;
}

static inline bool AddrIsInShadowGap(uptr a) {
  PROFILE_ASAN_MAPPING();
  if (kMidMemBeg) {
    if (a <= kShadowGapEnd)
      return SHADOW_OFFSET == 0 || a >= kShadowGapBeg;
    return (a >= kShadowGap2Beg && a <= kShadowGap2End) ||
           (a >= kShadowGap3Beg && a <= kShadowGap3End);
  }
  // In zero-based shadow mode we treat addresses near zero as addresses
  // in shadow gap as well.
  if (SHADOW_OFFSET == 0)
    return a <= kShadowGapEnd;
  return a >= kShadowGapBeg && a <= kShadowGapEnd;
}

}  // namespace __asan

#endif

namespace __asan {

static inline uptr MemToShadowSize(uptr size) { return size >> SHADOW_SCALE; }

static inline bool AddrIsInMem(uptr a) {
  PROFILE_ASAN_MAPPING();
  return AddrIsInLowMem(a) || AddrIsInMidMem(a) || AddrIsInHighMem(a) ||
      (flags()->protect_shadow_gap == 0 && AddrIsInShadowGap(a));
}

static inline uptr MemToShadow(uptr p) {
  PROFILE_ASAN_MAPPING();
  CHECK(AddrIsInMem(p));
  return MEM_TO_SHADOW(p);
}

static inline bool AddrIsInShadow(uptr a) {
  PROFILE_ASAN_MAPPING();
  return AddrIsInLowShadow(a) || AddrIsInMidShadow(a) || AddrIsInHighShadow(a);
}

static inline bool AddrIsAlignedByGranularity(uptr a) {
  PROFILE_ASAN_MAPPING();
  return (a & (SHADOW_GRANULARITY - 1)) == 0;
}

static inline bool AddressIsPoisoned(uptr a) {
  PROFILE_ASAN_MAPPING();
  const uptr kAccessSize = 1;
  u8 *shadow_address = (u8*)MEM_TO_SHADOW(a);
  s8 shadow_value = *shadow_address;
  if (shadow_value) {
    u8 last_accessed_byte = (a & (SHADOW_GRANULARITY - 1))
                                 + kAccessSize - 1;
    return (last_accessed_byte >= shadow_value);
  }
  return false;
}

// Must be after all calls to PROFILE_ASAN_MAPPING().
static const uptr kAsanMappingProfileSize = __LINE__;

}  // namespace __asan

#endif  // ASAN_MAPPING_H
