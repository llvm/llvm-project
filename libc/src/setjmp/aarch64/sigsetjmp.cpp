//===-- Implementation of sigsetjmp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/setjmp/sigsetjmp.h"
#include "hdr/offsetof_macros.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/setjmp/setjmp_impl.h"
#include "src/setjmp/sigsetjmp_epilogue.h"

namespace LIBC_NAMESPACE_DECL {
[[gnu::naked]]
LLVM_LIBC_FUNCTION(int, sigsetjmp, (sigjmp_buf, int)) {
  asm(R"(
      cbz w1, %c[setjmp]

      str x30, [x0, %c[retaddr]]
      str x19, [x0, %c[extra]]
      mov x19, x0
      bl %c[setjmp]

      mov w1, w0
      mov x0, x19
      ldr x30, [x0, %c[retaddr]]
      ldr x19, [x0, %c[extra]]
      b %c[epilogue])" ::[retaddr] "i"(offsetof(__jmp_buf, sig_retaddr)),
      [extra] "i"(offsetof(__jmp_buf, sig_extra)), [setjmp] "i"(setjmp),
      [epilogue] "i"(sigsetjmp_epilogue)
      : "x0", "x1", "x19", "x30");
}
} // namespace LIBC_NAMESPACE_DECL
