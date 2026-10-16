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
LLVM_LIBC_FUNCTION(int, sigsetjmp, (sigjmp_buf buf)) {
#if defined(__thumb__) && __ARM_ARCH_ISA_THUMB == 1
  // Thumb1 does not support the high registers > r7 in stmia, so move them
  // into lower GPRs first.
  asm(R"(
      tst r1, r1
      bne .Ldosave
      b %c[setjmp]
.Ldosave:
      str r4, [r0, #%c[extra]]
      mov r4, lr
      str r4, [r0, #%c[retaddr]]
      mov r4, r0
      bl %c[setjmp]
      mov r1, r0
      mov r0, r4
      ldr r4, [r0, #%c[retaddr]]
      mov lr, r4
      ldr r4, [r0, #%c[extra]]
      b %c[epilogue]
  )" ::[retaddr] "i"(offsetof(__jmp_buf, sig_retaddr)),
      [extra] "i"(offsetof(__jmp_buf, sig_extra)), [setjmp] "i"(setjmp),
      [epilogue] "i"(sigsetjmp_epilogue)
      : "r0", "r1", "r4");
#else
  // Some thumb2 linkers do not support conditional branch to PLT.
  // We branch to local labels instead.
  asm(R"(
      tst r1, r1
      bne .Ldosave
      b %c[setjmp]
.Ldosave:
      str r4, [r0, #%c[extra]]
      str lr, [r0, #%c[retaddr]]
      mov r4, r0
      bl %c[setjmp]
      mov r1, r0
      mov r0, r4
      ldr lr, [r0, #%c[retaddr]]
      ldr r4, [r0, #%c[extra]]
      b %c[epilogue]
  )" ::[retaddr] "i"(offsetof(__jmp_buf, sig_retaddr)),
      [extra] "i"(offsetof(__jmp_buf, sig_extra)), [setjmp] "X"(setjmp),
      [epilogue] "X"(sigsetjmp_epilogue)
      : "r0", "r1", "r4");
#endif
}
} // namespace LIBC_NAMESPACE_DECL
