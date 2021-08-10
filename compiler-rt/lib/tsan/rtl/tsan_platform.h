//===-- tsan_platform.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
// Platform-specific code.
//===----------------------------------------------------------------------===//

#ifndef TSAN_PLATFORM_H
#define TSAN_PLATFORM_H

#if !defined(__LP64__) && !defined(_WIN64)
# error "Only 64-bit is supported"
#endif

#include "tsan_defs.h"
#include "tsan_trace.h"

namespace __tsan {

enum {
  // App memory is not mapped onto shadow memory range.
  kBrokenMapping = 1 << 0,
  // Mapping app memory and back does not produce the same address,
  // this can lead to wrong addresses in reports and potentially
  // other bad consequences.
  kBrokenReverseMapping = 1 << 1,
  // Mapping is non-linear for linear user range.
  // This is bad and can lead to unpredictable memory corruptions, etc
  // because range access functions assume linearity.
  kBrokenLinearity = 1 << 2,
};

/*
C/C++ on linux/x86_64 and freebsd/x86_64
0000 0000 1000 - 0080 0000 0000: main binary and/or MAP_32BIT mappings (512GB)
0040 0000 0000 - 0100 0000 0000: -
0100 0000 0000 - 2000 0000 0000: shadow
2000 0000 0000 - 3000 0000 0000: -
3000 0000 0000 - 4000 0000 0000: metainfo (memory blocks and sync objects)
4000 0000 0000 - 5500 0000 0000: -
5500 0000 0000 - 5680 0000 0000: pie binaries without ASLR or on 4.1+ kernels
5680 0000 0000 - 6000 0000 0000: -
6000 0000 0000 - 6200 0000 0000: traces
6200 0000 0000 - 7d00 0000 0000: -
7b00 0000 0000 - 7c00 0000 0000: heap
7c00 0000 0000 - 7e80 0000 0000: -
7e80 0000 0000 - 8000 0000 0000: modules and main thread stack

C/C++ on netbsd/amd64 can reuse the same mapping:
 * The address space starts from 0x1000 (option with 0x0) and ends with
   0x7f7ffffff000.
 * LoAppMem-kHeapMemEnd can be reused as it is.
 * No VDSO support.
 * No MidAppMem region.
 * No additional HeapMem region.
 * HiAppMem contains the stack, loader, shared libraries and heap.
 * Stack on NetBSD/amd64 has prereserved 128MB.
 * Heap grows downwards (top-down).
 * ASLR must be disabled per-process or globally.
*/
struct Mapping48AddressSpace {
  static const uptr kMetaShadowBeg = 0x300000000000ull;
  static const uptr kMetaShadowEnd = 0x340000000000ull;
  static const uptr kTraceMemBeg   = 0x600000000000ull;
  static const uptr kTraceMemEnd   = 0x620000000000ull;
  static const uptr kShadowBeg     = 0x010000000000ull;
  static const uptr kShadowEnd     = 0x200000000000ull;
  static const uptr kHeapMemBeg    = 0x7b0000000000ull;
  static const uptr kHeapMemEnd    = 0x7c0000000000ull;
  static const uptr kLoAppMemBeg   = 0x000000001000ull;
  static const uptr kLoAppMemEnd   = 0x008000000000ull;
  static const uptr kMidAppMemBeg  = 0x550000000000ull;
  static const uptr kMidAppMemEnd  = 0x568000000000ull;
  static const uptr kHiAppMemBeg   = 0x7e8000000000ull;
  static const uptr kHiAppMemEnd   = 0x800000000000ull;
  static const uptr kShadowMsk = 0x780000000000ull;
  static const uptr kShadowXor = 0x040000000000ull;
  static const uptr kShadowAdd = 0x000000000000ull;
  static const uptr kVdsoBeg       = 0xf000000000000000ull;
};

/*
C/C++ on linux/mips64 (40-bit VMA)
0000 0000 00 - 0100 0000 00: -                                           (4 GB)
0100 0000 00 - 0200 0000 00: main binary                                 (4 GB)
0200 0000 00 - 2000 0000 00: -                                         (120 GB)
2000 0000 00 - 4000 0000 00: shadow                                    (128 GB)
4000 0000 00 - 5000 0000 00: metainfo (memory blocks and sync objects)  (64 GB)
5000 0000 00 - aa00 0000 00: -                                         (360 GB)
aa00 0000 00 - ab00 0000 00: main binary (PIE)                           (4 GB)
ab00 0000 00 - b000 0000 00: -                                          (20 GB)
b000 0000 00 - b200 0000 00: traces                                      (8 GB)
b200 0000 00 - fe00 0000 00: -                                         (304 GB)
fe00 0000 00 - ff00 0000 00: heap                                        (4 GB)
ff00 0000 00 - ff80 0000 00: -                                           (2 GB)
ff80 0000 00 - ffff ffff ff: modules and main thread stack              (<2 GB)
*/
struct MappingMips64_40 {
  static const uptr kMetaShadowBeg = 0x4000000000ull;
  static const uptr kMetaShadowEnd = 0x5000000000ull;
  static const uptr kTraceMemBeg   = 0xb000000000ull;
  static const uptr kTraceMemEnd   = 0xb200000000ull;
  static const uptr kShadowBeg     = 0x2000000000ull;
  static const uptr kShadowEnd     = 0x4000000000ull;
  static const uptr kHeapMemBeg    = 0xfe00000000ull;
  static const uptr kHeapMemEnd    = 0xff00000000ull;
  static const uptr kLoAppMemBeg   = 0x0100000000ull;
  static const uptr kLoAppMemEnd   = 0x0200000000ull;
  static const uptr kMidAppMemBeg  = 0xaa00000000ull;
  static const uptr kMidAppMemEnd  = 0xab00000000ull;
  static const uptr kHiAppMemBeg   = 0xff80000000ull;
  static const uptr kHiAppMemEnd   = 0xffffffffffull;
  static const uptr kShadowMsk = 0xf800000000ull;
  static const uptr kShadowXor = 0x0800000000ull;
  static const uptr kShadowAdd = 0x0000000000ull;
  static const uptr kVdsoBeg       = 0xfffff00000ull;
};

/*
C/C++ on Darwin/iOS/ARM64 (36-bit VMA, 64 GB VM)
0000 0000 00 - 0100 0000 00: -                                    (4 GB)
0100 0000 00 - 0200 0000 00: main binary, modules, thread stacks  (4 GB)
0200 0000 00 - 0300 0000 00: heap                                 (4 GB)
0300 0000 00 - 0400 0000 00: -                                    (4 GB)
0400 0000 00 - 0c00 0000 00: shadow memory                       (32 GB)
0c00 0000 00 - 0d00 0000 00: -                                    (4 GB)
0d00 0000 00 - 0e00 0000 00: metainfo                             (4 GB)
0e00 0000 00 - 0f00 0000 00: -                                    (4 GB)
0f00 0000 00 - 0fc0 0000 00: traces                               (3 GB)
0fc0 0000 00 - 1000 0000 00: -
*/
struct MappingAppleAarch64 {
  static const uptr kLoAppMemBeg   = 0x0100000000ull;
  static const uptr kLoAppMemEnd   = 0x0200000000ull;
  static const uptr kHeapMemBeg    = 0x0200000000ull;
  static const uptr kHeapMemEnd    = 0x0300000000ull;
  static const uptr kShadowBeg     = 0x0400000000ull;
  static const uptr kShadowEnd     = 0x0c00000000ull;
  static const uptr kMetaShadowBeg = 0x0d00000000ull;
  static const uptr kMetaShadowEnd = 0x0e00000000ull;
  static const uptr kTraceMemBeg   = 0x0f00000000ull;
  static const uptr kTraceMemEnd   = 0x0fc0000000ull;
  static const uptr kHiAppMemBeg   = 0x0fc0000000ull;
  static const uptr kHiAppMemEnd   = 0x0fc0000000ull;
  static const uptr kShadowMsk = 0x0ull;
  static const uptr kShadowXor = 0x0ull;
  static const uptr kShadowAdd = 0x0ull;
  static const uptr kVdsoBeg       = 0x7000000000000000ull;
  static const uptr kMidAppMemBeg = 0;
  static const uptr kMidAppMemEnd = 0;
};

/*
C/C++ on linux/aarch64 (39-bit VMA)
0000 0010 00 - 0100 0000 00: main binary
0100 0000 00 - 0800 0000 00: -
0800 0000 00 - 2000 0000 00: shadow memory
2000 0000 00 - 3100 0000 00: -
3100 0000 00 - 3400 0000 00: metainfo
3400 0000 00 - 5500 0000 00: -
5500 0000 00 - 5600 0000 00: main binary (PIE)
5600 0000 00 - 6000 0000 00: -
6000 0000 00 - 6200 0000 00: traces
6200 0000 00 - 7d00 0000 00: -
7c00 0000 00 - 7d00 0000 00: heap
7d00 0000 00 - 7fff ffff ff: modules and main thread stack
*/
struct MappingAarch64_39 {
  static const uptr kLoAppMemBeg   = 0x0000001000ull;
  static const uptr kLoAppMemEnd   = 0x0100000000ull;
  static const uptr kShadowBeg     = 0x0800000000ull;
  static const uptr kShadowEnd     = 0x2000000000ull;
  static const uptr kMetaShadowBeg = 0x3100000000ull;
  static const uptr kMetaShadowEnd = 0x3400000000ull;
  static const uptr kMidAppMemBeg  = 0x5500000000ull;
  static const uptr kMidAppMemEnd  = 0x5600000000ull;
  static const uptr kTraceMemBeg   = 0x6000000000ull;
  static const uptr kTraceMemEnd   = 0x6200000000ull;
  static const uptr kHeapMemBeg    = 0x7c00000000ull;
  static const uptr kHeapMemEnd    = 0x7d00000000ull;
  static const uptr kHiAppMemBeg   = 0x7e00000000ull;
  static const uptr kHiAppMemEnd   = 0x7fffffffffull;
  static const uptr kShadowMsk = 0x7800000000ull;
  static const uptr kShadowXor = 0x0200000000ull;
  static const uptr kShadowAdd = 0x0000000000ull;
  static const uptr kVdsoBeg       = 0x7f00000000ull;
};

/*
C/C++ on linux/aarch64 (42-bit VMA)
00000 0010 00 - 01000 0000 00: main binary
01000 0000 00 - 10000 0000 00: -
10000 0000 00 - 20000 0000 00: shadow memory
20000 0000 00 - 26000 0000 00: -
26000 0000 00 - 28000 0000 00: metainfo
28000 0000 00 - 2aa00 0000 00: -
2aa00 0000 00 - 2ab00 0000 00: main binary (PIE)
2ab00 0000 00 - 36200 0000 00: -
36200 0000 00 - 36240 0000 00: traces
36240 0000 00 - 3e000 0000 00: -
3e000 0000 00 - 3f000 0000 00: heap
3f000 0000 00 - 3ffff ffff ff: modules and main thread stack
*/
struct MappingAarch64_42 {
  static const uptr kBroken = kBrokenReverseMapping;
  static const uptr kLoAppMemBeg   = 0x00000001000ull;
  static const uptr kLoAppMemEnd   = 0x01000000000ull;
  static const uptr kShadowBeg     = 0x10000000000ull;
  static const uptr kShadowEnd     = 0x20000000000ull;
  static const uptr kMetaShadowBeg = 0x26000000000ull;
  static const uptr kMetaShadowEnd = 0x28000000000ull;
  static const uptr kMidAppMemBeg  = 0x2aa00000000ull;
  static const uptr kMidAppMemEnd  = 0x2ab00000000ull;
  static const uptr kTraceMemBeg   = 0x36200000000ull;
  static const uptr kTraceMemEnd   = 0x36400000000ull;
  static const uptr kHeapMemBeg    = 0x3e000000000ull;
  static const uptr kHeapMemEnd    = 0x3f000000000ull;
  static const uptr kHiAppMemBeg   = 0x3f000000000ull;
  static const uptr kHiAppMemEnd   = 0x3ffffffffffull;
  static const uptr kShadowMsk = 0x3c000000000ull;
  static const uptr kShadowXor = 0x04000000000ull;
  static const uptr kShadowAdd = 0x00000000000ull;
  static const uptr kVdsoBeg       = 0x37f00000000ull;
};

struct MappingAarch64_48 {
  static const uptr kLoAppMemBeg   = 0x0000000001000ull;
  static const uptr kLoAppMemEnd   = 0x0000200000000ull;
  static const uptr kShadowBeg     = 0x0002000000000ull;
  static const uptr kShadowEnd     = 0x0004000000000ull;
  static const uptr kMetaShadowBeg = 0x0005000000000ull;
  static const uptr kMetaShadowEnd = 0x0006000000000ull;
  static const uptr kMidAppMemBeg  = 0x0aaaa00000000ull;
  static const uptr kMidAppMemEnd  = 0x0aaaf00000000ull;
  static const uptr kTraceMemBeg   = 0x0f06000000000ull;
  static const uptr kTraceMemEnd   = 0x0f06200000000ull;
  static const uptr kHeapMemBeg    = 0x0ffff00000000ull;
  static const uptr kHeapMemEnd    = 0x0ffff00000000ull;
  static const uptr kHiAppMemBeg   = 0x0ffff00000000ull;
  static const uptr kHiAppMemEnd   = 0x1000000000000ull;
  static const uptr kShadowMsk = 0x0fff800000000ull;
  static const uptr kShadowXor = 0x0000800000000ull;
  static const uptr kShadowAdd = 0x0000000000000ull;
  static const uptr kVdsoBeg       = 0xffff000000000ull;
};

/*
C/C++ on linux/powerpc64 (44-bit VMA)
0000 0000 0100 - 0001 0000 0000: main binary
0001 0000 0000 - 0001 0000 0000: -
0001 0000 0000 - 0b00 0000 0000: shadow
0b00 0000 0000 - 0b00 0000 0000: -
0b00 0000 0000 - 0d00 0000 0000: metainfo (memory blocks and sync objects)
0d00 0000 0000 - 0d00 0000 0000: -
0d00 0000 0000 - 0f00 0000 0000: traces
0f00 0000 0000 - 0f00 0000 0000: -
0f00 0000 0000 - 0f50 0000 0000: heap
0f50 0000 0000 - 0f60 0000 0000: -
0f60 0000 0000 - 1000 0000 0000: modules and main thread stack
*/
struct MappingPPC64_44 {
  static const uptr kBroken =
      kBrokenMapping | kBrokenReverseMapping | kBrokenLinearity;
  static const uptr kMetaShadowBeg = 0x0b0000000000ull;
  static const uptr kMetaShadowEnd = 0x0d0000000000ull;
  static const uptr kTraceMemBeg   = 0x0d0000000000ull;
  static const uptr kTraceMemEnd   = 0x0f0000000000ull;
  static const uptr kShadowBeg     = 0x000100000000ull;
  static const uptr kShadowEnd     = 0x0b0000000000ull;
  static const uptr kLoAppMemBeg   = 0x000000000100ull;
  static const uptr kLoAppMemEnd   = 0x000100000000ull;
  static const uptr kHeapMemBeg    = 0x0f0000000000ull;
  static const uptr kHeapMemEnd    = 0x0f5000000000ull;
  static const uptr kHiAppMemBeg   = 0x0f6000000000ull;
  static const uptr kHiAppMemEnd   = 0x100000000000ull; // 44 bits
  static const uptr kShadowMsk = 0x0f0000000000ull;
  static const uptr kShadowXor = 0x002100000000ull;
  static const uptr kShadowAdd = 0x000000000000ull;
  static const uptr kVdsoBeg       = 0x3c0000000000000ull;
  static const uptr kMidAppMemBeg = 0;
  static const uptr kMidAppMemEnd = 0;
};

/*
C/C++ on linux/powerpc64 (46-bit VMA)
0000 0000 1000 - 0100 0000 0000: main binary
0100 0000 0000 - 0200 0000 0000: -
0100 0000 0000 - 1000 0000 0000: shadow
1000 0000 0000 - 1000 0000 0000: -
1000 0000 0000 - 2000 0000 0000: metainfo (memory blocks and sync objects)
2000 0000 0000 - 2000 0000 0000: -
2000 0000 0000 - 2200 0000 0000: traces
2200 0000 0000 - 3d00 0000 0000: -
3d00 0000 0000 - 3e00 0000 0000: heap
3e00 0000 0000 - 3e80 0000 0000: -
3e80 0000 0000 - 4000 0000 0000: modules and main thread stack
*/
struct MappingPPC64_46 {
  static const uptr kMetaShadowBeg = 0x100000000000ull;
  static const uptr kMetaShadowEnd = 0x200000000000ull;
  static const uptr kTraceMemBeg   = 0x200000000000ull;
  static const uptr kTraceMemEnd   = 0x220000000000ull;
  static const uptr kShadowBeg     = 0x010000000000ull;
  static const uptr kShadowEnd     = 0x100000000000ull;
  static const uptr kHeapMemBeg    = 0x3d0000000000ull;
  static const uptr kHeapMemEnd    = 0x3e0000000000ull;
  static const uptr kLoAppMemBeg   = 0x000000001000ull;
  static const uptr kLoAppMemEnd   = 0x010000000000ull;
  static const uptr kHiAppMemBeg   = 0x3e8000000000ull;
  static const uptr kHiAppMemEnd   = 0x400000000000ull; // 46 bits
  static const uptr kShadowMsk = 0x3c0000000000ull;
  static const uptr kShadowXor = 0x020000000000ull;
  static const uptr kShadowAdd = 0x000000000000ull;
  static const uptr kVdsoBeg       = 0x7800000000000000ull;
  static const uptr kMidAppMemBeg = 0;
  static const uptr kMidAppMemEnd = 0;
};

/*
C/C++ on linux/powerpc64 (47-bit VMA)
0000 0000 1000 - 0100 0000 0000: main binary
0100 0000 0000 - 0200 0000 0000: -
0100 0000 0000 - 1000 0000 0000: shadow
1000 0000 0000 - 1000 0000 0000: -
1000 0000 0000 - 2000 0000 0000: metainfo (memory blocks and sync objects)
2000 0000 0000 - 2000 0000 0000: -
2000 0000 0000 - 2200 0000 0000: traces
2200 0000 0000 - 7d00 0000 0000: -
7d00 0000 0000 - 7e00 0000 0000: heap
7e00 0000 0000 - 7e80 0000 0000: -
7e80 0000 0000 - 8000 0000 0000: modules and main thread stack
*/
struct MappingPPC64_47 {
  static const uptr kMetaShadowBeg = 0x100000000000ull;
  static const uptr kMetaShadowEnd = 0x200000000000ull;
  static const uptr kTraceMemBeg   = 0x200000000000ull;
  static const uptr kTraceMemEnd   = 0x220000000000ull;
  static const uptr kShadowBeg     = 0x010000000000ull;
  static const uptr kShadowEnd     = 0x100000000000ull;
  static const uptr kHeapMemBeg    = 0x7d0000000000ull;
  static const uptr kHeapMemEnd    = 0x7e0000000000ull;
  static const uptr kLoAppMemBeg   = 0x000000001000ull;
  static const uptr kLoAppMemEnd   = 0x010000000000ull;
  static const uptr kHiAppMemBeg   = 0x7e8000000000ull;
  static const uptr kHiAppMemEnd   = 0x800000000000ull; // 47 bits
  static const uptr kShadowMsk = 0x7c0000000000ull;
  static const uptr kShadowXor = 0x020000000000ull;
  static const uptr kShadowAdd = 0x000000000000ull;
  static const uptr kVdsoBeg       = 0x7800000000000000ull;
  static const uptr kMidAppMemBeg = 0;
  static const uptr kMidAppMemEnd = 0;
};

/*
C/C++ on linux/s390x
While the kernel provides a 64-bit address space, we have to restrict ourselves
to 48 bits due to how e.g. SyncVar::GetId() works.
0000 0000 1000 - 0e00 0000 0000: binary, modules, stacks - 14 TiB
0e00 0000 0000 - 4000 0000 0000: -
4000 0000 0000 - 8000 0000 0000: shadow - 64TiB (4 * app)
8000 0000 0000 - 9000 0000 0000: -
9000 0000 0000 - 9800 0000 0000: metainfo - 8TiB (0.5 * app)
9800 0000 0000 - a000 0000 0000: -
a000 0000 0000 - b000 0000 0000: traces - 16TiB (max history * 128k threads)
b000 0000 0000 - be00 0000 0000: -
be00 0000 0000 - c000 0000 0000: heap - 2TiB (max supported by the allocator)
*/
struct MappingS390x {
  static const uptr kMetaShadowBeg = 0x900000000000ull;
  static const uptr kMetaShadowEnd = 0x980000000000ull;
  static const uptr kTraceMemBeg   = 0xa00000000000ull;
  static const uptr kTraceMemEnd   = 0xb00000000000ull;
  static const uptr kShadowBeg     = 0x400000000000ull;
  static const uptr kShadowEnd     = 0x800000000000ull;
  static const uptr kHeapMemBeg    = 0xbe0000000000ull;
  static const uptr kHeapMemEnd    = 0xc00000000000ull;
  static const uptr kLoAppMemBeg   = 0x000000001000ull;
  static const uptr kLoAppMemEnd   = 0x0e0000000000ull;
  static const uptr kHiAppMemBeg   = 0xc00000004000ull;
  static const uptr kHiAppMemEnd   = 0xc00000004000ull;
  static const uptr kShadowMsk = 0xb00000000000ull;
  static const uptr kShadowXor = 0x100000000000ull;
  static const uptr kShadowAdd = 0x000000000000ull;
  static const uptr kVdsoBeg       = 0xfffffffff000ull;
  static const uptr kMidAppMemBeg = 0;
  static const uptr kMidAppMemEnd = 0;
};

/* Go on linux, darwin and freebsd on x86_64
0000 0000 1000 - 0000 1000 0000: executable
0000 1000 0000 - 00c0 0000 0000: -
00c0 0000 0000 - 00e0 0000 0000: heap
00e0 0000 0000 - 2000 0000 0000: -
2000 0000 0000 - 2380 0000 0000: shadow
2380 0000 0000 - 3000 0000 0000: -
3000 0000 0000 - 4000 0000 0000: metainfo (memory blocks and sync objects)
4000 0000 0000 - 6000 0000 0000: -
6000 0000 0000 - 6200 0000 0000: traces
6200 0000 0000 - 8000 0000 0000: -
*/

struct MappingGo48 {
  static const uptr kMetaShadowBeg = 0x300000000000ull;
  static const uptr kMetaShadowEnd = 0x400000000000ull;
  static const uptr kTraceMemBeg   = 0x600000000000ull;
  static const uptr kTraceMemEnd   = 0x620000000000ull;
  static const uptr kShadowBeg     = 0x200000000000ull;
  static const uptr kShadowEnd     = 0x238000000000ull;
  static const uptr kLoAppMemBeg = 0x000000001000ull;
  static const uptr kLoAppMemEnd = 0x00e000000000ull;
  static const uptr kMidAppMemBeg = 0;
  static const uptr kMidAppMemEnd = 0;
  static const uptr kHiAppMemBeg = 0;
  static const uptr kHiAppMemEnd = 0;
  static const uptr kHeapMemBeg = 0;
  static const uptr kHeapMemEnd = 0;
  static const uptr kVdsoBeg = 0;
  static const uptr kShadowMsk = 0;
  static const uptr kShadowXor = 0;
  static const uptr kShadowAdd = 0x200000000000ull;
};

/* Go on windows
0000 0000 1000 - 0000 1000 0000: executable
0000 1000 0000 - 00f8 0000 0000: -
00c0 0000 0000 - 00e0 0000 0000: heap
00e0 0000 0000 - 0100 0000 0000: -
0100 0000 0000 - 0500 0000 0000: shadow
0500 0000 0000 - 0700 0000 0000: traces
0700 0000 0000 - 0770 0000 0000: metainfo (memory blocks and sync objects)
07d0 0000 0000 - 8000 0000 0000: -
*/

struct MappingGoWindows {
  static const uptr kMetaShadowBeg = 0x070000000000ull;
  static const uptr kMetaShadowEnd = 0x077000000000ull;
  static const uptr kTraceMemBeg = 0x050000000000ull;
  static const uptr kTraceMemEnd = 0x070000000000ull;
  static const uptr kShadowBeg     = 0x010000000000ull;
  static const uptr kShadowEnd     = 0x050000000000ull;
  static const uptr kLoAppMemBeg = 0x000000001000ull;
  static const uptr kLoAppMemEnd = 0x00e000000000ull;
  static const uptr kMidAppMemBeg = 0;
  static const uptr kMidAppMemEnd = 0;
  static const uptr kHiAppMemBeg = 0;
  static const uptr kHiAppMemEnd = 0;
  static const uptr kHeapMemBeg = 0;
  static const uptr kHeapMemEnd = 0;
  static const uptr kVdsoBeg = 0;
  static const uptr kShadowMsk = 0;
  static const uptr kShadowXor = 0;
  static const uptr kShadowAdd = 0x010000000000ull;
};

/* Go on linux/powerpc64 (46-bit VMA)
0000 0000 1000 - 0000 1000 0000: executable
0000 1000 0000 - 00c0 0000 0000: -
00c0 0000 0000 - 00e0 0000 0000: heap
00e0 0000 0000 - 2000 0000 0000: -
2000 0000 0000 - 2380 0000 0000: shadow
2380 0000 0000 - 2400 0000 0000: -
2400 0000 0000 - 3400 0000 0000: metainfo (memory blocks and sync objects)
3400 0000 0000 - 3600 0000 0000: -
3600 0000 0000 - 3800 0000 0000: traces
3800 0000 0000 - 4000 0000 0000: -
*/

struct MappingGoPPC64_46 {
  static const uptr kMetaShadowBeg = 0x240000000000ull;
  static const uptr kMetaShadowEnd = 0x340000000000ull;
  static const uptr kTraceMemBeg   = 0x360000000000ull;
  static const uptr kTraceMemEnd   = 0x380000000000ull;
  static const uptr kShadowBeg     = 0x200000000000ull;
  static const uptr kShadowEnd     = 0x238000000000ull;
  static const uptr kLoAppMemBeg = 0x000000001000ull;
  static const uptr kLoAppMemEnd = 0x00e000000000ull;
  static const uptr kMidAppMemBeg = 0;
  static const uptr kMidAppMemEnd = 0;
  static const uptr kHiAppMemBeg = 0;
  static const uptr kHiAppMemEnd = 0;
  static const uptr kHeapMemBeg = 0;
  static const uptr kHeapMemEnd = 0;
  static const uptr kVdsoBeg = 0;
  static const uptr kShadowMsk = 0;
  static const uptr kShadowXor = 0;
  static const uptr kShadowAdd = 0x200000000000ull;
};

/* Go on linux/powerpc64 (47-bit VMA)
0000 0000 1000 - 0000 1000 0000: executable
0000 1000 0000 - 00c0 0000 0000: -
00c0 0000 0000 - 00e0 0000 0000: heap
00e0 0000 0000 - 2000 0000 0000: -
2000 0000 0000 - 3000 0000 0000: shadow
3000 0000 0000 - 3000 0000 0000: -
3000 0000 0000 - 4000 0000 0000: metainfo (memory blocks and sync objects)
4000 0000 0000 - 6000 0000 0000: -
6000 0000 0000 - 6200 0000 0000: traces
6200 0000 0000 - 8000 0000 0000: -
*/

struct MappingGoPPC64_47 {
  static const uptr kMetaShadowBeg = 0x300000000000ull;
  static const uptr kMetaShadowEnd = 0x400000000000ull;
  static const uptr kTraceMemBeg   = 0x600000000000ull;
  static const uptr kTraceMemEnd   = 0x620000000000ull;
  static const uptr kShadowBeg     = 0x200000000000ull;
  static const uptr kShadowEnd     = 0x300000000000ull;
  static const uptr kLoAppMemBeg = 0x000000001000ull;
  static const uptr kLoAppMemEnd = 0x00e000000000ull;
  static const uptr kMidAppMemBeg = 0;
  static const uptr kMidAppMemEnd = 0;
  static const uptr kHiAppMemBeg = 0;
  static const uptr kHiAppMemEnd = 0;
  static const uptr kHeapMemBeg = 0;
  static const uptr kHeapMemEnd = 0;
  static const uptr kVdsoBeg = 0;
  static const uptr kShadowMsk = 0;
  static const uptr kShadowXor = 0;
  static const uptr kShadowAdd = 0x200000000000ull;
};

/* Go on linux/aarch64 (48-bit VMA) and darwin/aarch64 (47-bit VMA)
0000 0000 1000 - 0000 1000 0000: executable
0000 1000 0000 - 00c0 0000 0000: -
00c0 0000 0000 - 00e0 0000 0000: heap
00e0 0000 0000 - 2000 0000 0000: -
2000 0000 0000 - 3000 0000 0000: shadow
3000 0000 0000 - 3000 0000 0000: -
3000 0000 0000 - 4000 0000 0000: metainfo (memory blocks and sync objects)
4000 0000 0000 - 6000 0000 0000: -
6000 0000 0000 - 6200 0000 0000: traces
6200 0000 0000 - 8000 0000 0000: -
*/
struct MappingGoAarch64 {
  static const uptr kMetaShadowBeg = 0x300000000000ull;
  static const uptr kMetaShadowEnd = 0x400000000000ull;
  static const uptr kTraceMemBeg   = 0x600000000000ull;
  static const uptr kTraceMemEnd   = 0x620000000000ull;
  static const uptr kShadowBeg     = 0x200000000000ull;
  static const uptr kShadowEnd     = 0x300000000000ull;
  static const uptr kLoAppMemBeg = 0x000000001000ull;
  static const uptr kLoAppMemEnd = 0x00e000000000ull;
  static const uptr kMidAppMemBeg = 0;
  static const uptr kMidAppMemEnd = 0;
  static const uptr kHiAppMemBeg = 0;
  static const uptr kHiAppMemEnd = 0;
  static const uptr kHeapMemBeg = 0;
  static const uptr kHeapMemEnd = 0;
  static const uptr kVdsoBeg = 0;
  static const uptr kShadowMsk = 0;
  static const uptr kShadowXor = 0;
  static const uptr kShadowAdd = 0x200000000000ull;
};

/*
Go on linux/mips64 (47-bit VMA)
0000 0000 1000 - 0000 1000 0000: executable
0000 1000 0000 - 00c0 0000 0000: -
00c0 0000 0000 - 00e0 0000 0000: heap
00e0 0000 0000 - 2000 0000 0000: -
2000 0000 0000 - 3000 0000 0000: shadow
3000 0000 0000 - 3000 0000 0000: -
3000 0000 0000 - 4000 0000 0000: metainfo (memory blocks and sync objects)
4000 0000 0000 - 6000 0000 0000: -
6000 0000 0000 - 6200 0000 0000: traces
6200 0000 0000 - 8000 0000 0000: -
*/
struct MappingGoMips64_47 {
  static const uptr kMetaShadowBeg = 0x300000000000ull;
  static const uptr kMetaShadowEnd = 0x400000000000ull;
  static const uptr kTraceMemBeg = 0x600000000000ull;
  static const uptr kTraceMemEnd = 0x620000000000ull;
  static const uptr kShadowBeg = 0x200000000000ull;
  static const uptr kShadowEnd = 0x300000000000ull;
  static const uptr kLoAppMemBeg = 0x000000001000ull;
  static const uptr kLoAppMemEnd = 0x00e000000000ull;
  static const uptr kMidAppMemBeg = 0;
  static const uptr kMidAppMemEnd = 0;
  static const uptr kHiAppMemBeg = 0;
  static const uptr kHiAppMemEnd = 0;
  static const uptr kHeapMemBeg = 0;
  static const uptr kHeapMemEnd = 0;
  static const uptr kVdsoBeg = 0;
  static const uptr kShadowMsk = 0;
  static const uptr kShadowXor = 0;
  static const uptr kShadowAdd = 0x200000000000ull;
};

/*
Go on linux/s390x
0000 0000 1000 - 1000 0000 0000: executable and heap - 16 TiB
1000 0000 0000 - 4000 0000 0000: -
4000 0000 0000 - 8000 0000 0000: shadow - 64TiB (4 * app)
8000 0000 0000 - 9000 0000 0000: -
9000 0000 0000 - 9800 0000 0000: metainfo - 8TiB (0.5 * app)
9800 0000 0000 - a000 0000 0000: -
a000 0000 0000 - b000 0000 0000: traces - 16TiB (max history * 128k threads)
*/
struct MappingGoS390x {
  static const uptr kMetaShadowBeg = 0x900000000000ull;
  static const uptr kMetaShadowEnd = 0x980000000000ull;
  static const uptr kTraceMemBeg   = 0xa00000000000ull;
  static const uptr kTraceMemEnd   = 0xb00000000000ull;
  static const uptr kShadowBeg     = 0x400000000000ull;
  static const uptr kShadowEnd     = 0x800000000000ull;
  static const uptr kLoAppMemBeg = 0x000000001000ull;
  static const uptr kLoAppMemEnd = 0x100000000000ull;
  static const uptr kMidAppMemBeg = 0;
  static const uptr kMidAppMemEnd = 0;
  static const uptr kHiAppMemBeg = 0;
  static const uptr kHiAppMemEnd = 0;
  static const uptr kHeapMemBeg = 0;
  static const uptr kHeapMemEnd = 0;
  static const uptr kVdsoBeg = 0;
  static const uptr kShadowMsk = 0;
  static const uptr kShadowXor = 0;
  static const uptr kShadowAdd = 0x400000000000ull;
};

extern uptr vmaSize;

template <typename Func, typename Arg>
ALWAYS_INLINE auto SelectMapping(Arg arg) {
#if SANITIZER_GO
#  if defined(__powerpc64__)
  switch (vmaSize) {
    case 46:
      return Func::template Apply<MappingGoPPC64_46>(arg);
    case 47:
      return Func::template Apply<MappingGoPPC64_47>(arg);
  }
#  elif defined(__mips64)
  return Func::template Apply<MappingGoMips64_47>(arg);
#  elif defined(__s390x__)
  return Func::template Apply<MappingGoS390x>(arg);
#  elif defined(__aarch64__)
  return Func::template Apply<MappingGoAarch64>(arg);
#  elif SANITIZER_WINDOWS
  return Func::template Apply<MappingGoWindows>(arg);
#  else
  return Func::template Apply<MappingGo48>(arg);
#  endif
#else  // SANITIZER_GO
#  if defined(__x86_64__) || defined(SANITIZER_IOSSIM) || \
      SANITIZER_MAC && !SANITIZER_IOS
  return Func::template Apply<Mapping48AddressSpace>(arg);
#  elif defined(__aarch64__) && defined(__APPLE__)
  return Func::template Apply<MappingAppleAarch64>(arg);
#  elif defined(__aarch64__) && !defined(__APPLE__)
  switch (vmaSize) {
    case 39:
      return Func::template Apply<MappingAarch64_39>(arg);
    case 42:
      return Func::template Apply<MappingAarch64_42>(arg);
    case 48:
      return Func::template Apply<MappingAarch64_48>(arg);
  }
#  elif defined(__powerpc64__)
  switch (vmaSize) {
    case 44:
      return Func::template Apply<MappingPPC64_44>(arg);
    case 46:
      return Func::template Apply<MappingPPC64_46>(arg);
    case 47:
      return Func::template Apply<MappingPPC64_47>(arg);
  }
#  elif defined(__mips64)
  return Func::template Apply<MappingMips64_40>(arg);
#  elif defined(__s390x__)
  return Func::template Apply<MappingS390x>(arg);
#  else
#    error "unsupported platform"
#  endif
#endif
  Die();
}

template <typename Func>
void ForEachMapping() {
  Func::template Apply<Mapping48AddressSpace>();
  Func::template Apply<MappingMips64_40>();
  Func::template Apply<MappingAppleAarch64>();
  Func::template Apply<MappingAarch64_39>();
  Func::template Apply<MappingAarch64_42>();
  Func::template Apply<MappingAarch64_48>();
  Func::template Apply<MappingPPC64_44>();
  Func::template Apply<MappingPPC64_46>();
  Func::template Apply<MappingPPC64_47>();
  Func::template Apply<MappingS390x>();
  Func::template Apply<MappingGo48>();
  Func::template Apply<MappingGoWindows>();
  Func::template Apply<MappingGoPPC64_46>();
  Func::template Apply<MappingGoPPC64_47>();
  Func::template Apply<MappingGoAarch64>();
  Func::template Apply<MappingGoMips64_47>();
  Func::template Apply<MappingGoS390x>();
}

enum MappingType {
  MAPPING_LO_APP_BEG,
  MAPPING_LO_APP_END,
  MAPPING_HI_APP_BEG,
  MAPPING_HI_APP_END,
  MAPPING_MID_APP_BEG,
  MAPPING_MID_APP_END,
  MAPPING_HEAP_BEG,
  MAPPING_HEAP_END,
  MAPPING_SHADOW_BEG,
  MAPPING_SHADOW_END,
  MAPPING_META_SHADOW_BEG,
  MAPPING_META_SHADOW_END,
  MAPPING_TRACE_BEG,
  MAPPING_TRACE_END,
  MAPPING_VDSO_BEG,
};

struct MappingField {
  template <typename Mapping>
  static uptr Apply(MappingType type) {
    switch (type) {
    case MAPPING_LO_APP_BEG: return Mapping::kLoAppMemBeg;
    case MAPPING_LO_APP_END: return Mapping::kLoAppMemEnd;
    case MAPPING_MID_APP_BEG: return Mapping::kMidAppMemBeg;
    case MAPPING_MID_APP_END: return Mapping::kMidAppMemEnd;
    case MAPPING_HI_APP_BEG: return Mapping::kHiAppMemBeg;
    case MAPPING_HI_APP_END: return Mapping::kHiAppMemEnd;
    case MAPPING_HEAP_BEG: return Mapping::kHeapMemBeg;
    case MAPPING_HEAP_END: return Mapping::kHeapMemEnd;
    case MAPPING_VDSO_BEG: return Mapping::kVdsoBeg;
    case MAPPING_SHADOW_BEG: return Mapping::kShadowBeg;
    case MAPPING_SHADOW_END: return Mapping::kShadowEnd;
    case MAPPING_META_SHADOW_BEG: return Mapping::kMetaShadowBeg;
    case MAPPING_META_SHADOW_END: return Mapping::kMetaShadowEnd;
    case MAPPING_TRACE_BEG: return Mapping::kTraceMemBeg;
    case MAPPING_TRACE_END: return Mapping::kTraceMemEnd;
    }
    Die();
  }
};

ALWAYS_INLINE
uptr LoAppMemBeg(void) {
  return SelectMapping<MappingField>(MAPPING_LO_APP_BEG);
}
ALWAYS_INLINE
uptr LoAppMemEnd(void) {
  return SelectMapping<MappingField>(MAPPING_LO_APP_END);
}

ALWAYS_INLINE
uptr MidAppMemBeg(void) {
  return SelectMapping<MappingField>(MAPPING_MID_APP_BEG);
}
ALWAYS_INLINE
uptr MidAppMemEnd(void) {
  return SelectMapping<MappingField>(MAPPING_MID_APP_END);
}

ALWAYS_INLINE
uptr HeapMemBeg(void) { return SelectMapping<MappingField>(MAPPING_HEAP_BEG); }
ALWAYS_INLINE
uptr HeapMemEnd(void) { return SelectMapping<MappingField>(MAPPING_HEAP_END); }

ALWAYS_INLINE
uptr HiAppMemBeg(void) {
  return SelectMapping<MappingField>(MAPPING_HI_APP_BEG);
}
ALWAYS_INLINE
uptr HiAppMemEnd(void) {
  return SelectMapping<MappingField>(MAPPING_HI_APP_END);
}

ALWAYS_INLINE
uptr VdsoBeg(void) { return SelectMapping<MappingField>(MAPPING_VDSO_BEG); }

ALWAYS_INLINE
uptr ShadowBeg(void) { return SelectMapping<MappingField>(MAPPING_SHADOW_BEG); }
ALWAYS_INLINE
uptr ShadowEnd(void) { return SelectMapping<MappingField>(MAPPING_SHADOW_END); }

ALWAYS_INLINE
uptr MetaShadowBeg(void) {
  return SelectMapping<MappingField>(MAPPING_META_SHADOW_BEG);
}
ALWAYS_INLINE
uptr MetaShadowEnd(void) {
  return SelectMapping<MappingField>(MAPPING_META_SHADOW_END);
}

ALWAYS_INLINE
uptr TraceMemBeg(void) {
  return SelectMapping<MappingField>(MAPPING_TRACE_BEG);
}
ALWAYS_INLINE
uptr TraceMemEnd(void) {
  return SelectMapping<MappingField>(MAPPING_TRACE_END);
}

struct IsAppMemImpl {
  template <typename Mapping>
  static bool Apply(uptr mem) {
  return (mem >= Mapping::kHeapMemBeg && mem < Mapping::kHeapMemEnd) ||
         (mem >= Mapping::kMidAppMemBeg && mem < Mapping::kMidAppMemEnd) ||
         (mem >= Mapping::kLoAppMemBeg && mem < Mapping::kLoAppMemEnd) ||
         (mem >= Mapping::kHiAppMemBeg && mem < Mapping::kHiAppMemEnd);
  }
};

ALWAYS_INLINE
bool IsAppMem(uptr mem) { return SelectMapping<IsAppMemImpl>(mem); }

struct IsShadowMemImpl {
  template <typename Mapping>
  static bool Apply(uptr mem) {
    return mem >= Mapping::kShadowBeg && mem <= Mapping::kShadowEnd;
  }
};

ALWAYS_INLINE
bool IsShadowMem(RawShadow *p) {
  return SelectMapping<IsShadowMemImpl>(reinterpret_cast<uptr>(p));
}

struct IsMetaMemImpl {
  template <typename Mapping>
  static bool Apply(uptr mem) {
    return mem >= Mapping::kMetaShadowBeg && mem <= Mapping::kMetaShadowEnd;
  }
};

ALWAYS_INLINE
bool IsMetaMem(const u32 *p) {
  return SelectMapping<IsMetaMemImpl>(reinterpret_cast<uptr>(p));
}

struct MemToShadowImpl {
  template <typename Mapping>
  static uptr Apply(uptr x) {
    DCHECK(IsAppMemImpl::Apply<Mapping>(x));
    return (((x) & ~(Mapping::kShadowMsk | (kShadowCell - 1))) ^
            Mapping::kShadowXor) *
               kShadowCnt +
           Mapping::kShadowAdd;
  }
};

ALWAYS_INLINE
RawShadow *MemToShadow(uptr x) {
  return reinterpret_cast<RawShadow *>(SelectMapping<MemToShadowImpl>(x));
}

struct MemToMetaImpl {
  template <typename Mapping>
  static u32 *Apply(uptr x) {
    DCHECK(IsAppMemImpl::Apply<Mapping>(x));
    return (u32 *)(((((x) & ~(Mapping::kShadowMsk | (kMetaShadowCell - 1)))) /
                    kMetaShadowCell * kMetaShadowSize) |
                   Mapping::kMetaShadowBeg);
  }
};

ALWAYS_INLINE
u32 *MemToMeta(uptr x) { return SelectMapping<MemToMetaImpl>(x); }

struct ShadowToMemImpl {
  template <typename Mapping>
  static uptr Apply(uptr sp) {
    if (!IsShadowMemImpl::Apply<Mapping>(sp))
      return 0;
    // The shadow mapping is non-linear and we've lost some bits, so we don't
    // have an easy way to restore the original app address. But the mapping is
    // a bijection, so we try to restore the address as belonging to
    // low/mid/high range consecutively and see if shadow->app->shadow mapping
    // gives us the same address.
    uptr p = ((sp - Mapping::kShadowAdd) / kShadowCnt) ^ Mapping::kShadowXor;
    if (p >= Mapping::kLoAppMemBeg && p < Mapping::kLoAppMemEnd &&
        MemToShadowImpl::Apply<Mapping>(p) == sp)
      return p;
    if (Mapping::kMidAppMemBeg) {
      uptr p_mid = p + (Mapping::kMidAppMemBeg & Mapping::kShadowMsk);
      if (p_mid >= Mapping::kMidAppMemBeg && p_mid < Mapping::kMidAppMemEnd &&
          MemToShadowImpl::Apply<Mapping>(p_mid) == sp)
        return p_mid;
    }
    return p | Mapping::kShadowMsk;
  }
};

ALWAYS_INLINE
uptr ShadowToMem(RawShadow *s) {
  return SelectMapping<ShadowToMemImpl>(reinterpret_cast<uptr>(s));
}

// The additional page is to catch shadow stack overflow as paging fault.
// Windows wants 64K alignment for mmaps.
const uptr kTotalTraceSize = (kTraceSize * sizeof(Event) + sizeof(Trace)
    + (64 << 10) + (64 << 10) - 1) & ~((64 << 10) - 1);

struct GetThreadTraceImpl {
  template <typename Mapping>
  static uptr Apply(uptr tid) {
    uptr p = Mapping::kTraceMemBeg + tid * kTotalTraceSize;
    DCHECK_LT(p, Mapping::kTraceMemEnd);
    return p;
  }
};

ALWAYS_INLINE
uptr GetThreadTrace(int tid) { return SelectMapping<GetThreadTraceImpl>(tid); }

struct GetThreadTraceHeaderImpl {
  template <typename Mapping>
  static uptr Apply(uptr tid) {
    uptr p = Mapping::kTraceMemBeg + tid * kTotalTraceSize +
             kTraceSize * sizeof(Event);
    DCHECK_LT(p, Mapping::kTraceMemEnd);
    return p;
  }
};

ALWAYS_INLINE
uptr GetThreadTraceHeader(int tid) {
  return SelectMapping<GetThreadTraceHeaderImpl>(tid);
}

void InitializePlatform();
void InitializePlatformEarly();
void CheckAndProtect();
void InitializeShadowMemoryPlatform();
void FlushShadowMemory();
void WriteMemoryProfile(char *buf, uptr buf_size, uptr nthread, uptr nlive);
int ExtractResolvFDs(void *state, int *fds, int nfd);
int ExtractRecvmsgFDs(void *msg, int *fds, int nfd);
uptr ExtractLongJmpSp(uptr *env);
void ImitateTlsWrite(ThreadState *thr, uptr tls_addr, uptr tls_size);

int call_pthread_cancel_with_cleanup(int (*fn)(void *arg),
                                     void (*cleanup)(void *arg), void *arg);

void DestroyThreadState();
void PlatformCleanUpThreadState(ThreadState *thr);

}  // namespace __tsan

#endif  // TSAN_PLATFORM_H
