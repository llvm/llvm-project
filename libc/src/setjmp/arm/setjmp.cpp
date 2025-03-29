//===-- Implementation of setjmp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/setjmp/setjmp_impl.h"

namespace LIBC_NAMESPACE_DECL {

#if defined(__thumb__) && __ARM_ARCH_ISA_THUMB == 1

[[gnu::naked, gnu::target("thumb")]] LLVM_LIBC_FUNCTION(int, setjmp,
                                                        (jmp_buf buf)) {
  asm(R"(
      # Store r4, r5, r6, and r7 into buf.
      stmia r0!, {r4-r7}

      # Store r8, r9, r10, r11, sp, and lr into buf. Thumb(1) doesn't support
      # the high registers > r7 in stmia, so move them into lower GPRs first.
      # Thumb(1) also doesn't support using str with sp or lr, move them
      # together with the rest.
      mov r1, r8
      mov r2, r9
      mov r3, r10
      stmia r0!, {r1-r3}

      mov r1, r11
      mov r2, sp
      mov r3, lr
      stmia r0!, {r1-r3}

      # Return 0.
      movs r0, #0
      bx lr)");
}

#else // Thumb2 or ARM

// TODO(https://github.com/llvm/llvm-project/issues/94061): fp registers
// (d0-d16)
// TODO(https://github.com/llvm/llvm-project/issues/94062): pac+bti
[[gnu::naked]] LLVM_LIBC_FUNCTION(int, setjmp, (jmp_buf buf)) {
  asm(R"(
      # While sp may appear in a register list for ARM mode, it may not for
      # Thumb2 mode. Just move it into r12 then stm that, so that this code
      # is portable between ARM and Thumb2.
      mov r12, sp

      # Store r4, r5, r6, r7, r8, r9, r10, r11, sp, and lr into buf.
      stm r0, {r4-r12, lr}

      # Return zero.
      mov r0, #0
      bx lr)");
}

#endif

} // namespace LIBC_NAMESPACE_DECL
