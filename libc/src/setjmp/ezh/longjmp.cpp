//===-- Implementation of longjmp for EZH ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/setjmp/longjmp.h"

namespace LIBC_NAMESPACE_DECL {

[[gnu::naked]] LLVM_LIBC_FUNCTION(void, longjmp, (jmp_buf buf, int val)) {
  asm(R"(
      # Restore r4, r5, r6, r7, sp, and ra from buf (r0)
      e_ldr r4, r0, 0
      e_ldr r5, r0, 4
      e_ldr r6, r0, 8
      e_ldr r7, r0, 12
      e_ldr sp, r0, 16
      e_ldr ra, r0, 20

      # Move val (passed in r1) to return register r0
      e_mov r0, r1

      # If val is 0, longjmp must return 1 instead
      e_sub_imms r1, r0, 0
      e_goto_nz .Ljump_back
      e_load_imm r0, 1

  .Ljump_back:
      e_goto ra
  )");
}

} // namespace LIBC_NAMESPACE_DECL
