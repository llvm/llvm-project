//===-- Implementation of setjmp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/common.h"
#include "src/setjmp/setjmp_impl.h"

#if !defined(LIBC_TARGET_ARCH_IS_ARM)
#error "Invalid file include"
#endif

namespace LIBC_NAMESPACE {

[[gnu::naked]]
LLVM_LIBC_FUNCTION(int, setjmp, (__jmp_buf * buf)) {
  asm("mov r12, sp\n\t"
      "stm.w r0!, {r4, r5, r6, r7, r8, r9, r10, r11, r12, lr}\n\t"
      "mov.w r0, 0\n\t"
      "bx lr");
}

} // namespace LIBC_NAMESPACE
