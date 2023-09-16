//===-- Hexagon implementation of longjmp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/setjmp/longjmp.h"
#include "src/__support/common.h"
#include "src/__support/macros/properties/architectures.h"

#include <setjmp.h>

#if !defined(LIBC_TARGET_ARCH_IS_HEXAGON)
#error "Invalid file include"
#endif

#define LOAD(reg, val) LIBC_INLINE_ASM(#reg " = memd(%0)\n\t" : : "r"(&val) :)

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(void, longjmp, (__jmp_buf * buf, int val)) {
  LOAD(r17:16, buf->__regs[0]);
  LOAD(r19:18, buf->__regs[2]);
  LOAD(r21:20, buf->__regs[4]);
  LOAD(r23:22, buf->__regs[6]);
  LOAD(r25:24, buf->__regs[8]);
  LOAD(r27:26, buf->__regs[10]);
  LOAD(r29:28, buf->__regs[12]);
  LOAD(r31:30, buf->__regs[14]);

  val = val == 0 ? 1 : val;
  LIBC_INLINE_ASM("r0 = %0\n\t" : : "r"(val) :);
}

} // namespace LIBC_NAMESPACE
