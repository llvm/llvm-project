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

// https://www.ibm.com/docs/en/aix/7.3?topic=concepts-system-memory-allocation-using-malloc-subsystem
//
// For 64-bit on AIX,
// - Data, heap, bss region is from 0x0000 0001 0000 0000 to
//   0x07ff ffff ffff ffff (1ULL << 59).
// - Shared library regions is from:
//      0x0900 0000 0000 0000 to 0x09ff ffff ffff ffff
//   or 0x0800 0000 0000 0000 to 0x08ff ffff ffff ffff ((1ULL << 52) * 2)
// - mmap region is from 0x0a00 0000 0000 0000 to 0x0aff ffff ffff ffff
//   (1ULL << 52).
// - Initial stack region is from 0x0f00 0000 0000 0000 to
//   0x0fff ffff ffff ffff (1ULL << 56).
//
// All above ranges are too big. And after verifying on AIX,(these datas are
// from experiments on AIX72, AIX OS may change this layout in future)
// - the biggest heap size is 1ULL << 47.
// - the biggest global variable size is 1ULL << 29. (Which may be put in shared
//   library data regions because global variables may be compiled to shared
//   libraries.)
//     the related address range for shared library data regions is:
//          0x0900 1000 0000 0000 to 0x0900 1001 0000 0000
//       or 0x0800 1000 0000 0000 to 0x0800 1001 0000 0000 (when above range is
//          used by system libraries.)
// - the biggest mmap size is 1ULL << 46.
// - the biggest stack size is 1ULL << 32.
//
// We don't need so big heap and mmap, calling mmap for shadow memory for such
// big heap and mmap is quite slow on AIX, so to balance runtime and examinable
// memory size, we use 1ULL << 39(512GB) as size for each region except mmap
// region. For mmap region, aix system mmap function may return a big range
// address, we allocate 1ULL << 41(2TB).
//
// So the reasonable user space region size is:
// - Data, heap, bss is from 0x0 to 0x0000 007f ffff ffff
// - Shared library data is from:
//        0x0900 1000 0000 0000 to 0x0900 107f ffff ffff
//     or 0x0800 1000 0000 0000 to 0x0800 107f ffff ffff
// - mmap is from 0x0a00 0000 0000 0000 to 0x0a00 01ff ffff ffff
// - Stack is from 0x0fff ff80 0000 0000 to 0x0fff ffff ffff ffff
//
// AIX64 set ASAN_SHADOW_OFFSET_CONST at 0x0a01000000000000 because mmap
// memory starts at 0x0a00000000000000 and shadow memory should be allocated
// there. And we keep 0x0a00000000000000 to 0x0a01000000000000 For user mmap
// usage.

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
