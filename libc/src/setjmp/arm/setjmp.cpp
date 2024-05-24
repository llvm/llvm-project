//===-- Implementation of setjmp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/common.h"
#include "src/setjmp/setjmp_impl.h"

namespace LIBC_NAMESPACE {

#if defined(__thumb__) && __ARM_ARCH_ISA_THUMB == 1

[[gnu::naked]]
LLVM_LIBC_FUNCTION(int, setjmp, (__jmp_buf * buf)) {
  asm(R"(
      # Store r4, r5, r6, and r7 into buf.
      stmia r0!, {r4-r7}

      # Store r8, r9, r10, and r11 into buf. Thumb(1) doesn't support the high
      # registers > r7 in stmia, so move them into lower GPRs first.
      mov r4, r8
      mov r5, r9
      mov r6, r10
      mov r7, r11
      stmia r0!, {r4-r7}

      # Store sp into buf. Thumb(1) doesn't support sp in str, move to GPR
      # first.
      mov r4, sp
      str r4, [r0]

      # Return 0.
      movs r0, #0
      bx lr)");
}

#else // Thumb2 or ARM

[[gnu::naked]]
LLVM_LIBC_FUNCTION(int, setjmp, (__jmp_buf * buf)) {
  asm(R"(
      mov r12, sp
      stm r0, {r4-r12, lr}
      mov r0, #0
      bx lr)");
}

#endif

} // namespace LIBC_NAMESPACE
