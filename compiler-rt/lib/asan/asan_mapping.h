//===-- asan_mapping.h ------------------------------------------*- C++ -*-===//
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
// Defines ASan memory mapping.
//===----------------------------------------------------------------------===//
#ifndef ASAN_MAPPING_H
#define ASAN_MAPPING_H

#include "asan_internal.h"

// The full explanation of the memory mapping could be found here:
// http://code.google.com/p/address-sanitizer/wiki/AddressSanitizerAlgorithm
//
// Typical shadow mapping on Linux/x86_64 with SHADOW_OFFSET == 0x00007fff8000:
// || `[0x10007fff8000, 0x7fffffffffff]` || HighMem    ||
// || `[0x02008fff7000, 0x10007fff7fff]` || HighShadow ||
// || `[0x00008fff7000, 0x02008fff6fff]` || ShadowGap  ||
// || `[0x00007fff8000, 0x00008fff6fff]` || LowShadow  ||
// || `[0x000000000000, 0x00007fff7fff]` || LowMem     ||
//
// When SHADOW_OFFSET is zero (-pie):
// || `[0x100000000000, 0x7fffffffffff]` || HighMem    ||
// || `[0x020000000000, 0x0fffffffffff]` || HighShadow ||
// || `[0x000000040000, 0x01ffffffffff]` || ShadowGap  ||
//
// Special case when something is already mapped between
// 0x003000000000 and 0x005000000000 (e.g. when prelink is installed):
// || `[0x10007fff8000, 0x7fffffffffff]` || HighMem    ||
// || `[0x02008fff7000, 0x10007fff7fff]` || HighShadow ||
// || `[0x005000000000, 0x02008fff6fff]` || ShadowGap3 ||
// || `[0x003000000000, 0x004fffffffff]` || MidMem     ||
// || `[0x000a7fff8000, 0x002fffffffff]` || ShadowGap2 ||
// || `[0x00067fff8000, 0x000a7fff7fff]` || MidShadow  ||
// || `[0x00008fff7000, 0x00067fff7fff]` || ShadowGap  ||
// || `[0x00007fff8000, 0x00008fff6fff]` || LowShadow  ||
// || `[0x000000000000, 0x00007fff7fff]` || LowMem     ||
//
// Default Linux/i386 mapping:
// || `[0x40000000, 0xffffffff]` || HighMem    ||
// || `[0x28000000, 0x3fffffff]` || HighShadow ||
// || `[0x24000000, 0x27ffffff]` || ShadowGap  ||
// || `[0x20000000, 0x23ffffff]` || LowShadow  ||
// || `[0x00000000, 0x1fffffff]` || LowMem     ||
//
// Default Linux/MIPS mapping:
// || `[0x2aaa8000, 0xffffffff]` || HighMem    ||
// || `[0x0fffd000, 0x2aaa7fff]` || HighShadow ||
// || `[0x0bffd000, 0x0fffcfff]` || ShadowGap  ||
// || `[0x0aaa8000, 0x0bffcfff]` || LowShadow  ||
// || `[0x00000000, 0x0aaa7fff]` || LowMem     ||

static const u64 kDefaultShadowScale = 3;
static const u64 kDefaultShadowOffset32 = 1ULL << 29;
static const u64 kDefaultShadowOffset64 = 1ULL << 44;
static const u64 kDefaultShort64bitShadowOffset = 0x7FFF8000;  // < 2G.
static const u64 kPPC64_ShadowOffset64 = 1ULL << 41;
static const u64 kMIPS32_ShadowOffset32 = 0x0aaa8000;

#if ASAN_FLEXIBLE_MAPPING_AND_OFFSET == 1
extern SANITIZER_INTERFACE_ATTRIBUTE uptr __asan_mapping_scale;
extern SANITIZER_INTERFACE_ATTRIBUTE uptr __asan_mapping_offset;
# define SHADOW_SCALE (__asan_mapping_scale)
# define SHADOW_OFFSET (__asan_mapping_offset)
#else
# define SHADOW_SCALE kDefaultShadowScale
# if SANITIZER_ANDROID
#  define SHADOW_OFFSET (0)
# else
#  if SANITIZER_WORDSIZE == 32
#   if defined(__mips__)
#     define SHADOW_OFFSET kMIPS32_ShadowOffset32
#   else
#     define SHADOW_OFFSET kDefaultShadowOffset32
#   endif
#  else
#   if defined(__powerpc64__)
#    define SHADOW_OFFSET kPPC64_ShadowOffset64
#   elif SANITIZER_MAC
#    define SHADOW_OFFSET kDefaultShadowOffset64
#   else
#    define SHADOW_OFFSET kDefaultShort64bitShadowOffset
#   endif
#  endif
# endif
#endif  // ASAN_FLEXIBLE_MAPPING_AND_OFFSET

#define SHADOW_GRANULARITY (1ULL << SHADOW_SCALE)
#define MEM_TO_SHADOW(mem) (((mem) >> SHADOW_SCALE) + (SHADOW_OFFSET))
#define SHADOW_TO_MEM(shadow) (((shadow) - SHADOW_OFFSET) << SHADOW_SCALE)

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
#define kZeroBaseShadowStart (1 << 18)

#define kShadowGapBeg   (kLowShadowEnd ? kLowShadowEnd + 1 \
                                       : kZeroBaseShadowStart)
#define kShadowGapEnd   ((kMidMemBeg ? kMidShadowBeg : kHighShadowBeg) - 1)

#define kShadowGap2Beg (kMidMemBeg ? kMidShadowEnd + 1 : 0)
#define kShadowGap2End (kMidMemBeg ? kMidMemBeg - 1 : 0)

#define kShadowGap3Beg (kMidMemBeg ? kMidMemEnd + 1 : 0)
#define kShadowGap3End (kMidMemBeg ? kHighShadowBeg - 1 : 0)

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

static inline bool AddrIsInLowMem(uptr a) {
  PROFILE_ASAN_MAPPING();
  return a < kLowMemEnd;
}

static inline bool AddrIsInLowShadow(uptr a) {
  PROFILE_ASAN_MAPPING();
  return a >= kLowShadowBeg && a <= kLowShadowEnd;
}

static inline bool AddrIsInHighMem(uptr a) {
  PROFILE_ASAN_MAPPING();
  return a >= kHighMemBeg && a <= kHighMemEnd;
}

static inline bool AddrIsInMidMem(uptr a) {
  PROFILE_ASAN_MAPPING();
  return kMidMemBeg && a >= kMidMemBeg && a <= kMidMemEnd;
}

static inline bool AddrIsInMem(uptr a) {
  PROFILE_ASAN_MAPPING();
  return AddrIsInLowMem(a) || AddrIsInMidMem(a) || AddrIsInHighMem(a);
}

static inline uptr MemToShadow(uptr p) {
  PROFILE_ASAN_MAPPING();
  CHECK(AddrIsInMem(p));
  return MEM_TO_SHADOW(p);
}

static inline bool AddrIsInHighShadow(uptr a) {
  PROFILE_ASAN_MAPPING();
  return a >= kHighShadowBeg && a <= kHighMemEnd;
}

static inline bool AddrIsInMidShadow(uptr a) {
  PROFILE_ASAN_MAPPING();
  return kMidMemBeg && a >= kMidShadowBeg && a <= kMidMemEnd;
}

static inline bool AddrIsInShadow(uptr a) {
  PROFILE_ASAN_MAPPING();
  return AddrIsInLowShadow(a) || AddrIsInMidShadow(a) || AddrIsInHighShadow(a);
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
