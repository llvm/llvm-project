
//===-- Implementation of longjmp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/setjmp/longjmp.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

#if defined(__thumb__) && __ARM_ARCH_ISA_THUMB == 1

[[gnu::naked, gnu::target("thumb")]] LLVM_LIBC_FUNCTION(void, longjmp,
                                                        (jmp_buf buf,
                                                         int val)) {
  asm(R"(
      # Reload r4, r5, r6, r7.
      ldmia r0!, {r4-r7}

      # Reload r8, r9. They cannot appear in register lists so load them
      # into the lower registers, then move them into place.
      ldmia r0!, {r2-r3}
      mov r8, r2
      mov r9, r3

      # Reload r10, r11. They cannot appear in register lists so load them
      # into the lower registers, then move them into place.
      ldmia r0!, {r2-r3}
      mov r10, r2
      mov r11, r3

      # Reload sp, lr. They cannot appear in register lists so load them
      # into the lower registers, then move them into place.
      ldmia r0!, {r2-r3}
      mov sp, r2
      mov lr, r3

      # return val ?: 1;
      movs r0, r1
      bne .Lret_val
      movs r0, #1

    .Lret_val:
      bx lr)");
}

#else // Thumb2 or ARM

// TODO(https://github.com/llvm/llvm-project/issues/94061): fp registers
// (d0-d16)
// TODO(https://github.com/llvm/llvm-project/issues/94062): pac+bti
[[gnu::naked]] LLVM_LIBC_FUNCTION(void, longjmp, (jmp_buf buf, int val)) {
  asm(R"(
      # While sp may appear in a register list for ARM mode, it may not for
      # Thumb2 mode. Just load the previous value of sp into r12 then move it
      # into sp, so that this code is portable between ARM and Thumb2.

      ldm r0, {r4-r12, lr}
      mov sp, r12

      # return val ?: 1;
      movs r0, r1
      it eq
      moveq r0, #1
      bx lr)");
}

#endif

} // namespace LIBC_NAMESPACE_DECL
