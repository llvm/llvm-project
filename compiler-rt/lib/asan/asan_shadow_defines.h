//===-- asan_shadow_defines.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// Defines ASan memory mapping used by assembly portion.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_platform.h"

// The full explanation of the memory mapping could be found here:
// https://github.com/google/sanitizers/wiki/AddressSanitizerAlgorithm
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
// Default Linux/i386 mapping on x86_64 machine:
// || `[0x40000000, 0xffffffff]` || HighMem    ||
// || `[0x28000000, 0x3fffffff]` || HighShadow ||
// || `[0x24000000, 0x27ffffff]` || ShadowGap  ||
// || `[0x20000000, 0x23ffffff]` || LowShadow  ||
// || `[0x00000000, 0x1fffffff]` || LowMem     ||
//
// Default Linux/i386 mapping on i386 machine
// (addresses starting with 0xc0000000 are reserved
// for kernel and thus not sanitized):
// || `[0x38000000, 0xbfffffff]` || HighMem    ||
// || `[0x27000000, 0x37ffffff]` || HighShadow ||
// || `[0x24000000, 0x26ffffff]` || ShadowGap  ||
// || `[0x20000000, 0x23ffffff]` || LowShadow  ||
// || `[0x00000000, 0x1fffffff]` || LowMem     ||
//
// Default Linux/MIPS32 mapping:
// || `[0x2aaa0000, 0xffffffff]` || HighMem    ||
// || `[0x0fff4000, 0x2aa9ffff]` || HighShadow ||
// || `[0x0bff4000, 0x0fff3fff]` || ShadowGap  ||
// || `[0x0aaa0000, 0x0bff3fff]` || LowShadow  ||
// || `[0x00000000, 0x0aa9ffff]` || LowMem     ||
//
// Default Linux/MIPS64 mapping:
// || `[0x4000000000, 0xffffffffff]` || HighMem    ||
// || `[0x2800000000, 0x3fffffffff]` || HighShadow ||
// || `[0x2400000000, 0x27ffffffff]` || ShadowGap  ||
// || `[0x2000000000, 0x23ffffffff]` || LowShadow  ||
// || `[0x0000000000, 0x1fffffffff]` || LowMem     ||
//
// Default Linux/RISCV64 Sv39 mapping:
// || `[0x1555550000, 0x3fffffffff]` || HighMem    ||
// || `[0x0fffffa000, 0x1555555fff]` || HighShadow ||
// || `[0x0effffa000, 0x0fffff9fff]` || ShadowGap  ||
// || `[0x0d55550000, 0x0effff9fff]` || LowShadow  ||
// || `[0x0000000000, 0x0d5554ffff]` || LowMem     ||
//
// Default Linux/AArch64 (39-bit VMA) mapping:
// || `[0x2000000000, 0x7fffffffff]` || highmem    ||
// || `[0x1400000000, 0x1fffffffff]` || highshadow ||
// || `[0x1200000000, 0x13ffffffff]` || shadowgap  ||
// || `[0x1000000000, 0x11ffffffff]` || lowshadow  ||
// || `[0x0000000000, 0x0fffffffff]` || lowmem     ||
//
// Default Linux/AArch64 (42-bit VMA) mapping:
// || `[0x10000000000, 0x3ffffffffff]` || highmem    ||
// || `[0x0a000000000, 0x0ffffffffff]` || highshadow ||
// || `[0x09000000000, 0x09fffffffff]` || shadowgap  ||
// || `[0x08000000000, 0x08fffffffff]` || lowshadow  ||
// || `[0x00000000000, 0x07fffffffff]` || lowmem     ||
//
// Default Linux/S390 mapping:
// || `[0x30000000, 0x7fffffff]` || HighMem    ||
// || `[0x26000000, 0x2fffffff]` || HighShadow ||
// || `[0x24000000, 0x25ffffff]` || ShadowGap  ||
// || `[0x20000000, 0x23ffffff]` || LowShadow  ||
// || `[0x00000000, 0x1fffffff]` || LowMem     ||
//
// Default Linux/SystemZ mapping:
// || `[0x14000000000000, 0x1fffffffffffff]` || HighMem    ||
// || `[0x12800000000000, 0x13ffffffffffff]` || HighShadow ||
// || `[0x12000000000000, 0x127fffffffffff]` || ShadowGap  ||
// || `[0x10000000000000, 0x11ffffffffffff]` || LowShadow  ||
// || `[0x00000000000000, 0x0fffffffffffff]` || LowMem     ||
//
// Default Linux/SPARC64 (52-bit VMA) mapping:
// || `[0x8000000000000, 0xfffffffffffff]` || HighMem    ||
// || `[0x1080000000000, 0x207ffffffffff]` || HighShadow ||
// || `[0x0090000000000, 0x107ffffffffff]` || ShadowGap  ||
// || `[0x0080000000000, 0x008ffffffffff]` || LowShadow  ||
// || `[0x0000000000000, 0x007ffffffffff]` || LowMem     ||
//
// Shadow mapping on FreeBSD/x86-64 with SHADOW_OFFSET == 0x400000000000:
// || `[0x500000000000, 0x7fffffffffff]` || HighMem    ||
// || `[0x4a0000000000, 0x4fffffffffff]` || HighShadow ||
// || `[0x480000000000, 0x49ffffffffff]` || ShadowGap  ||
// || `[0x400000000000, 0x47ffffffffff]` || LowShadow  ||
// || `[0x000000000000, 0x3fffffffffff]` || LowMem     ||
//
// Shadow mapping on FreeBSD/i386 with SHADOW_OFFSET == 0x40000000:
// || `[0x60000000, 0xffffffff]` || HighMem    ||
// || `[0x4c000000, 0x5fffffff]` || HighShadow ||
// || `[0x48000000, 0x4bffffff]` || ShadowGap  ||
// || `[0x40000000, 0x47ffffff]` || LowShadow  ||
// || `[0x00000000, 0x3fffffff]` || LowMem     ||
//
// Shadow mapping on NetBSD/x86-64 with SHADOW_OFFSET == 0x400000000000:
// || `[0x4feffffffe01, 0x7f7ffffff000]` || HighMem    ||
// || `[0x49fdffffffc0, 0x4feffffffe00]` || HighShadow ||
// || `[0x480000000000, 0x49fdffffffbf]` || ShadowGap  ||
// || `[0x400000000000, 0x47ffffffffff]` || LowShadow  ||
// || `[0x000000000000, 0x3fffffffffff]` || LowMem     ||
//
// Shadow mapping on NetBSD/i386 with SHADOW_OFFSET == 0x40000000:
// || `[0x60000000, 0xfffff000]` || HighMem    ||
// || `[0x4c000000, 0x5fffffff]` || HighShadow ||
// || `[0x48000000, 0x4bffffff]` || ShadowGap  ||
// || `[0x40000000, 0x47ffffff]` || LowShadow  ||
// || `[0x00000000, 0x3fffffff]` || LowMem     ||
//
// Default Windows/i386 mapping:
// (the exact location of HighShadow/HighMem may vary depending
//  on WoW64, /LARGEADDRESSAWARE, etc).
// || `[0x50000000, 0xffffffff]` || HighMem    ||
// || `[0x3a000000, 0x4fffffff]` || HighShadow ||
// || `[0x36000000, 0x39ffffff]` || ShadowGap  ||
// || `[0x30000000, 0x35ffffff]` || LowShadow  ||
// || `[0x00000000, 0x2fffffff]` || LowMem     ||

#if !defined(SANITIZER_WORDSIZE)
#  error "SANITIZER_WORDSIZE must be defined."
#endif

#if SANITIZER_FUCHSIA
#  define SHADOW_OFFSET_CONST 0x0
#elif SANITIZER_WORDSIZE == 32
#  if defined(__mips__)
#    define SHADOW_OFFSET_CONST 0xaaa0000
#  elif SANITIZER_FREEBSD
#    define SHADOW_OFFSET_CONST 0x40000000
#  elif SANITIZER_NETBSD
#    define SHADOW_OFFSET_CONST 0x40000000
#  elif SANITIZER_WINDOWS
#    define SHADOW_OFFSET_CONST 0x30000000
#  else
#    define SHADOW_OFFSET_CONST 0x20000000
#  endif
#elif SANITIZER_WORDSIZE == 64
#  if SANITIZER_RISCV64
#    define SHADOW_OFFSET_CONST 0x0d55550000
#  elif defined(__aarch64__)
#    define SHADOW_OFFSET_CONST 0x1000000000
#  elif defined(__powerpc64__)
#    define SHADOW_OFFSET_CONST 0x100000000000
#  elif defined(__s390x__)
#    define SHADOW_OFFSET_CONST 0x10000000000000
#  elif SANITIZER_FREEBSD
#    define SHADOW_OFFSET_CONST 0x400000000000
#  elif SANITIZER_NETBSD
#    define SHADOW_OFFSET_CONST 0x400000000000
#  elif SANITIZER_MAC
#    define SHADOW_OFFSET_CONST 0x100000000000
#  elif defined(__mips64)
#    define SHADOW_OFFSET_CONST 0x02000000000
#  elif defined(__sparc__)
#    define SHADOW_OFFSET_CONST 0x80000000000
#  else
#    if defined(ASAN_SHADOW_SCALE) && (ASAN_SHADOW_SCALE != 3)
#      error "Only ASAN_SHADOW_SCALE = 3 is supported."
#    endif
#    define SHADOW_OFFSET_CONST 0x7fff8000
#  endif
#endif
