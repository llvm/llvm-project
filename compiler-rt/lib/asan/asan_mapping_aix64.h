//===-- asan_mapping_aix64.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// AIX64-specific definitions for ASan memory mapping.
//===----------------------------------------------------------------------===//
#ifndef ASAN_MAPPING_AIX64_H
#define ASAN_MAPPING_AIX64_H

// 64-bit AIX process address space (AIX 7.1 and later):
//
// | Range                                   | Contents                     |
// |-----------------------------------------|------------------------------|
// | 0x0000000100000000 - 0x07ffffffffffffff | Program text, data and heap  |
// | 0x0800000000000000 - 0x08ffffffffffffff | Privately loaded objects     |
// | 0x0900000000000000 - 0x09ffffffffffffff | Shared library text and data |
// | 0x0a00000000000000 - 0x0affffffffffffff | mmap and shmat               |
// | 0x0f00000000000000 - 0x0fffffffffffffff | Initial stack                |
//
// Since AIX 7.1, the default address space allocator (esid_allocator=1)
// places mmap and shmat attachments without a fixed address in the
// 0x0a00000000000000 range; fixed-address attachments use 0x0700000000000000 -
// 0x07ffffffffffffff. See
// https://www.ibm.com/docs/en/aix/7.3?topic=concepts-system-memory-allocation-using-malloc-subsystem
// and section 1.2 of https://www.redbooks.ibm.com/redbooks/pdfs/sg247910.pdf.
//
// These ranges are spread over most of the 60-bit address space, so shadowing
// all of them would need a shadow too large to map, and reserving it would make
// startup slow. Programs only use a small part of each range in practice. On
// AIX 7.2, the largest observed sizes were:
// - heap: 1ULL << 47,
// - global variables: 1ULL << 29. These can be in shared library data, at
//   0x0900100000000000 - 0x0900100100000000, or at 0x0800100000000000 -
//   0x0800100100000000 when the first range is used by system libraries,
// - mmap: 1ULL << 46,
// - stack: 1ULL << 32.
//
// To balance runtime cost and coverage, ASan shadows 512GB (1ULL << 39) of
// each region, except mmap, which gets 2TB (1ULL << 41) because mmap can
// return large ranges. This gives the mapping below. The shadow itself is
// placed at 0x0a01000000000000 (ASAN_SHADOW_OFFSET_CONST), in the mmap region;
// 0x0a00000000000000 - 0x0a00ffffffffffff stays available for user mmap.

// NOTE: Users are not expected to use `mmap` specifying fixed address which is
// inside the shadow memory ranges.

// Default AIX64 mapping:
// || `[0x0fffff8000000000, 0x0fffffffffffffff]` || HighMem    ||
// || `[0x0a80fff000000000, 0x0a80ffffffffffff]` || HighShadow ||
// || `[0x0a41000000000000, 0x0a41003fffffffff]` || MidShadow  ||
// || `[0x0a21020000000000, 0x0a21020fffffffff]` || Mid2Shadow ||
// || `[0x0a01020000000000, 0x0a01020fffffffff]` || Mid3Shadow ||
// || `[0x0a01000000000000, 0x0a01000fffffffff]` || LowShadow  ||
// || `[0x0a00000000000000, 0x0a0001ffffffffff]` || MidMem     ||
// || `[0x0900100000000000, 0x0900107fffffffff]` || Mid2Mem    ||
// || `[0x0800100000000000, 0x0800107fffffffff]` || Mid3Mem    ||
// || `[0x0000000000000000, 0x0000007fffffffff]` || LowMem     ||

#define VMA_BITS 58
#define HIGH_BITS (64 - VMA_BITS)

#define MEM_TO_SHADOW(mem)                                       \
  ((((mem) << HIGH_BITS) >> (HIGH_BITS + (ASAN_SHADOW_SCALE))) + \
   ASAN_SHADOW_OFFSET)

#define SHADOW_TO_MEM(ptr) (__asan::ShadowToMemAIX64(ptr))

#define kLowMemBeg 0ULL
#define kLowMemEnd 0x0000007fffffffffULL

#define kLowShadowBeg ASAN_SHADOW_OFFSET
#define kLowShadowEnd MEM_TO_SHADOW(kLowMemEnd)

#define kHighMemBeg 0x0fffff8000000000ULL

#define kHighShadowBeg MEM_TO_SHADOW(kHighMemBeg)
#define kHighShadowEnd MEM_TO_SHADOW(kHighMemEnd)

#define kMidMemBeg 0x0a00000000000000ULL
#define kMidMemEnd 0x0a0001ffffffffffULL

#define kMidShadowBeg MEM_TO_SHADOW(kMidMemBeg)
#define kMidShadowEnd MEM_TO_SHADOW(kMidMemEnd)

#define kMid2MemBeg 0x0900100000000000ULL
#define kMid2MemEnd 0x0900107fffffffffULL

#define kMid2ShadowBeg MEM_TO_SHADOW(kMid2MemBeg)
#define kMid2ShadowEnd MEM_TO_SHADOW(kMid2MemEnd)

#define kMid3MemBeg 0x0800100000000000ULL
#define kMid3MemEnd 0x0800107fffffffffULL

#define kMid3ShadowBeg MEM_TO_SHADOW(kMid3MemBeg)
#define kMid3ShadowEnd MEM_TO_SHADOW(kMid3MemEnd)

// AIX does not care about the gaps.
#define kZeroBaseShadowStart 0
#define kZeroBaseMaxShadowStart 0

#define kShadowGapBeg 0
#define kShadowGapEnd 0

#define kShadowGap2Beg 0
#define kShadowGap2End 0

#define kShadowGap3Beg 0
#define kShadowGap3End 0

#define kShadowGap4Beg 0
#define kShadowGap4End 0

// The shadow is placed in the middle of the address space, not above all
// application memory.
#define kGaplessShadow (false)

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
  return (a >= kMidMemBeg && a <= kMidMemEnd) ||
         (a >= kMid2MemBeg && a <= kMid2MemEnd) ||
         (a >= kMid3MemBeg && a <= kMid3MemEnd);
}

static inline bool AddrIsInMidShadow(uptr a) {
  PROFILE_ASAN_MAPPING();
  return (a >= kMidShadowBeg && a <= kMidShadowEnd) ||
         (a >= kMid2ShadowBeg && a <= kMid2ShadowEnd) ||
         (a >= kMid3ShadowBeg && a <= kMid3ShadowEnd);
}

static inline bool AddrIsInHighMem(uptr a) {
  PROFILE_ASAN_MAPPING();
  return kHighMemBeg && a >= kHighMemBeg && a <= kHighMemEnd;
}

static inline bool AddrIsInHighShadow(uptr a) {
  PROFILE_ASAN_MAPPING();
  return kHighMemBeg && a >= kHighShadowBeg && a <= kHighShadowEnd;
}

static inline bool AddrIsInShadowGap(uptr a) { return false; }

static inline constexpr uptr ShadowToMemAIX64(uptr p) {
  PROFILE_ASAN_MAPPING();
  p -= ASAN_SHADOW_OFFSET;
  p <<= ASAN_SHADOW_SCALE;
  if (p >= 0x3ffff8000000000ULL) {
    // HighMem
    p |= (0x03ULL << VMA_BITS);
  } else if (p >= 0x100000000000ULL) {
    // MidShadow/Mid2Shadow/Mid2Shadow
    p |= (0x02ULL << VMA_BITS);
  }
  return p;
}

}  // namespace __asan

#endif  // ASAN_MAPPING_AIX64_H
