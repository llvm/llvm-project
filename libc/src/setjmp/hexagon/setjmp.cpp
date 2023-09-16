//===-- Hexagon implementation of setjmp ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/common.h"
#include "src/setjmp/setjmp_impl.h"

#include <setjmp.h>

#if !defined(LIBC_TARGET_ARCH_IS_HEXAGON)
#error "Invalid file include"
#endif

#define STORE(reg, val)                                                        \
  LIBC_INLINE_ASM("memd(%0) = " #reg "\n\t" : : "r"(&val) :)

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, setjmp, (__jmp_buf * buf)) {
  STORE(r17:16, buf->__regs[0]);
  STORE(r19:18, buf->__regs[2]);
  STORE(r21:20, buf->__regs[4]);
  STORE(r23:22, buf->__regs[6]);
  STORE(r25:24, buf->__regs[8]);
  STORE(r27:26, buf->__regs[10]);
  STORE(r29:28, buf->__regs[12]);
  STORE(r31:30, buf->__regs[14]);

  return 0;
}

} // namespace LIBC_NAMESPACE
