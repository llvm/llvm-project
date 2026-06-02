//===-- Implementation of setjmp for EZH ----------------------------------===//
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

[[gnu::naked]] LLVM_LIBC_FUNCTION(int, setjmp, (jmp_buf buf)) {
  asm(R"(
      # Store r4, r5, r6, r7, sp, and ra into buf (passed in r0)
      e_str r0, r4, 0
      e_str r0, r5, 4
      e_str r0, r6, 8
      e_str r0, r7, 12
      e_str r0, sp, 16
      e_str r0, ra, 20

      # Return 0
      e_load_imm r0, 0
      e_goto ra
  )");
}

} // namespace LIBC_NAMESPACE_DECL
