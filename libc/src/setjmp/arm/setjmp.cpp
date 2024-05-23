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

[[gnu::naked]]
LLVM_LIBC_FUNCTION(int, setjmp, (__jmp_buf * buf)) {
  asm(R"(
      mov r12, sp
      stm r0, {r4-r12, lr}
      mov r0, #0
      bx lr)");
}

} // namespace LIBC_NAMESPACE
