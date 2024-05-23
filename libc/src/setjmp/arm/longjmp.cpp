
//===-- Implementation of longjmp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/setjmp/longjmp.h"
#include "src/__support/common.h"
#include "src/__support/macros/properties/architectures.h"

#if !defined(LIBC_TARGET_ARCH_IS_ARM)
#error "Invalid file include"
#endif

namespace LIBC_NAMESPACE {

[[gnu::naked]]
LLVM_LIBC_FUNCTION(void, longjmp, (__jmp_buf * buf, int val)) {
  asm("ldm.w r0!, {r4, r5, r6, r7, r8, r9, r10, r11, r12, lr}\n\t"
      "mov sp, r12\n\t"
      "movs r0, r1\n\t"
      "it eq\n\t"
      "moveq r0, 1\n\t"
      "bx lr"
  );
}

} // namespace LIBC_NAMESPACE
