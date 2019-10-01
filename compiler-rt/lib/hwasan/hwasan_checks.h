//===-- hwasan_checks.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of HWAddressSanitizer.
//
//===----------------------------------------------------------------------===//

#ifndef HWASAN_CHECKS_H
#define HWASAN_CHECKS_H

#include "hwasan_mapping.h"
#include "sanitizer_common/sanitizer_common.h"

namespace __hwasan {
template <unsigned X>
__attribute__((always_inline)) static void SigTrap(uptr p) {
#if defined(__aarch64__)
  (void)p;
  // 0x900 is added to do not interfere with the kernel use of lower values of
  // brk immediate.
  register uptr x0 asm("x0") = p;
  asm("brk %1\n\t" ::"r"(x0), "n"(0x900 + X));
#elif defined(__x86_64__)
  // INT3 + NOP DWORD ptr [EAX + X] to pass X to our signal handler, 5 bytes
  // total. The pointer is passed via rdi.
  // 0x40 is added as a safeguard, to help distinguish our trap from others and
  // to avoid 0 offsets in the command (otherwise it'll be reduced to a
  // different nop command, the three bytes one).
  asm volatile(
      "int3\n"
      "nopl %c0(%%rax)\n" ::"n"(0x40 + X),
      "D"(p));
#else
  // FIXME: not always sigill.
  __builtin_trap();
#endif
  // __builtin_unreachable();
}

// Version with access size which is not power of 2
template <unsigned X>
__attribute__((always_inline)) static void SigTrap(uptr p, uptr size) {
#if defined(__aarch64__)
  register uptr x0 asm("x0") = p;
  register uptr x1 asm("x1") = size;
  asm("brk %2\n\t" ::"r"(x0), "r"(x1), "n"(0x900 + X));
#elif defined(__x86_64__)
  // Size is stored in rsi.
  asm volatile(
      "int3\n"
      "nopl %c0(%%rax)\n" ::"n"(0x40 + X),
      "D"(p), "S"(size));
#else
  __builtin_trap();
#endif
  // __builtin_unreachable();
}

enum class ErrorAction { Abort, Recover };
enum class AccessType { Load, Store };

template <ErrorAction EA, AccessType AT, unsigned LogSize>
__attribute__((always_inline, nodebug)) static void CheckAddress(uptr p) {
  tag_t ptr_tag = GetTagFromPointer(p);
  uptr ptr_raw = p & ~kAddressTagMask;
  tag_t mem_tag = *(tag_t *)MemToShadow(ptr_raw);
  if (UNLIKELY(ptr_tag != mem_tag)) {
    SigTrap<0x20 * (EA == ErrorAction::Recover) +
            0x10 * (AT == AccessType::Store) + LogSize>(p);
    if (EA == ErrorAction::Abort)
      __builtin_unreachable();
  }
}

template <ErrorAction EA, AccessType AT>
__attribute__((always_inline, nodebug)) static void CheckAddressSized(uptr p,
                                                                      uptr sz) {
  if (sz == 0)
    return;
  tag_t ptr_tag = GetTagFromPointer(p);
  uptr ptr_raw = p & ~kAddressTagMask;
  tag_t *shadow_first = (tag_t *)MemToShadow(ptr_raw);
  tag_t *shadow_last = (tag_t *)MemToShadow(ptr_raw + sz - 1);
  for (tag_t *t = shadow_first; t <= shadow_last; ++t)
    if (UNLIKELY(ptr_tag != *t)) {
      SigTrap<0x20 * (EA == ErrorAction::Recover) +
              0x10 * (AT == AccessType::Store) + 0xf>(p, sz);
      if (EA == ErrorAction::Abort)
        __builtin_unreachable();
    }
}
}  // end namespace __hwasan

#endif  // HWASAN_CHECKS_H
