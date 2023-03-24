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

#include <setjmp.h>

#if !defined(LIBC_TARGET_ARCH_IS_RISCV64)
#error "Invalid file include"
#endif

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(void, longjmp, (__jmp_buf * buf, int val)) {
  LIBC_INLINE_ASM("ld ra, %0\n\t" : : "m"(buf->__pc) :);
  LIBC_INLINE_ASM("ld s0, %0\n\t" : : "m"(buf->__regs[0]) :);
  LIBC_INLINE_ASM("ld s1, %0\n\t" : : "m"(buf->__regs[1]) :);
  LIBC_INLINE_ASM("ld s2, %0\n\t" : : "m"(buf->__regs[2]) :);
  LIBC_INLINE_ASM("ld s3, %0\n\t" : : "m"(buf->__regs[3]) :);
  LIBC_INLINE_ASM("ld s4, %0\n\t" : : "m"(buf->__regs[4]) :);
  LIBC_INLINE_ASM("ld s5, %0\n\t" : : "m"(buf->__regs[5]) :);
  LIBC_INLINE_ASM("ld s6, %0\n\t" : : "m"(buf->__regs[6]) :);
  LIBC_INLINE_ASM("ld s7, %0\n\t" : : "m"(buf->__regs[7]) :);
  LIBC_INLINE_ASM("ld s8, %0\n\t" : : "m"(buf->__regs[8]) :);
  LIBC_INLINE_ASM("ld s9, %0\n\t" : : "m"(buf->__regs[9]) :);
  LIBC_INLINE_ASM("ld s10, %0\n\t" : : "m"(buf->__regs[10]) :);
  LIBC_INLINE_ASM("ld s11, %0\n\t" : : "m"(buf->__regs[11]) :);
  LIBC_INLINE_ASM("ld sp, %0\n\t" : : "m"(buf->__sp) :);

#if __riscv_float_abi_double
  LIBC_INLINE_ASM("fld fs0, %0\n\t" : : "m"(buf->__fpregs[0]) :);
  LIBC_INLINE_ASM("fld fs1, %0\n\t" : : "m"(buf->__fpregs[1]) :);
  LIBC_INLINE_ASM("fld fs2, %0\n\t" : : "m"(buf->__fpregs[2]) :);
  LIBC_INLINE_ASM("fld fs3, %0\n\t" : : "m"(buf->__fpregs[3]) :);
  LIBC_INLINE_ASM("fld fs4, %0\n\t" : : "m"(buf->__fpregs[4]) :);
  LIBC_INLINE_ASM("fld fs5, %0\n\t" : : "m"(buf->__fpregs[5]) :);
  LIBC_INLINE_ASM("fld fs6, %0\n\t" : : "m"(buf->__fpregs[6]) :);
  LIBC_INLINE_ASM("fld fs7, %0\n\t" : : "m"(buf->__fpregs[7]) :);
  LIBC_INLINE_ASM("fld fs8, %0\n\t" : : "m"(buf->__fpregs[8]) :);
  LIBC_INLINE_ASM("fld fs9, %0\n\t" : : "m"(buf->__fpregs[9]) :);
  LIBC_INLINE_ASM("fld fs10, %0\n\t" : : "m"(buf->__fpregs[10]) :);
  LIBC_INLINE_ASM("fld fs11, %0\n\t" : : "m"(buf->__fpregs[11]) :);
#elif defined(__riscv_float_abi_single)
#error "longjmp implementation not available for the target architecture."
#endif

  val = val == 0 ? 1 : val;
  LIBC_INLINE_ASM("add a0, %0, zero\n\t" : : "r"(val) :);
}

} // namespace __llvm_libc
