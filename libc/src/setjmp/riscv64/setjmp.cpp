//===-- Implementation of setjmp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/common.h"
#include "src/setjmp/setjmp_impl.h"

#include <setjmp.h>

#if !defined(LIBC_TARGET_ARCH_IS_RISCV64)
#error "Invalid file include"
#endif

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, setjmp, (__jmp_buf * buf)) {
  LIBC_INLINE_ASM("sd ra, %0\n\t" : : "m"(buf->__pc) :);
  LIBC_INLINE_ASM("sd s0, %0\n\t" : : "m"(buf->__regs[0]) :);
  LIBC_INLINE_ASM("sd s1, %0\n\t" : : "m"(buf->__regs[1]) :);
  LIBC_INLINE_ASM("sd s2, %0\n\t" : : "m"(buf->__regs[2]) :);
  LIBC_INLINE_ASM("sd s3, %0\n\t" : : "m"(buf->__regs[3]) :);
  LIBC_INLINE_ASM("sd s4, %0\n\t" : : "m"(buf->__regs[4]) :);
  LIBC_INLINE_ASM("sd s5, %0\n\t" : : "m"(buf->__regs[5]) :);
  LIBC_INLINE_ASM("sd s6, %0\n\t" : : "m"(buf->__regs[6]) :);
  LIBC_INLINE_ASM("sd s7, %0\n\t" : : "m"(buf->__regs[7]) :);
  LIBC_INLINE_ASM("sd s8, %0\n\t" : : "m"(buf->__regs[8]) :);
  LIBC_INLINE_ASM("sd s9, %0\n\t" : : "m"(buf->__regs[9]) :);
  LIBC_INLINE_ASM("sd s10, %0\n\t" : : "m"(buf->__regs[10]) :);
  LIBC_INLINE_ASM("sd s11, %0\n\t" : : "m"(buf->__regs[11]) :);
  LIBC_INLINE_ASM("sd sp, %0\n\t" : : "m"(buf->__sp) :);

#if __riscv_float_abi_double
  LIBC_INLINE_ASM("fsd fs0, %0\n\t" : : "m"(buf->__fpregs[0]) :);
  LIBC_INLINE_ASM("fsd fs1, %0\n\t" : : "m"(buf->__fpregs[1]) :);
  LIBC_INLINE_ASM("fsd fs2, %0\n\t" : : "m"(buf->__fpregs[2]) :);
  LIBC_INLINE_ASM("fsd fs3, %0\n\t" : : "m"(buf->__fpregs[3]) :);
  LIBC_INLINE_ASM("fsd fs4, %0\n\t" : : "m"(buf->__fpregs[4]) :);
  LIBC_INLINE_ASM("fsd fs5, %0\n\t" : : "m"(buf->__fpregs[5]) :);
  LIBC_INLINE_ASM("fsd fs6, %0\n\t" : : "m"(buf->__fpregs[6]) :);
  LIBC_INLINE_ASM("fsd fs7, %0\n\t" : : "m"(buf->__fpregs[7]) :);
  LIBC_INLINE_ASM("fsd fs8, %0\n\t" : : "m"(buf->__fpregs[8]) :);
  LIBC_INLINE_ASM("fsd fs9, %0\n\t" : : "m"(buf->__fpregs[9]) :);
  LIBC_INLINE_ASM("fsd fs10, %0\n\t" : : "m"(buf->__fpregs[10]) :);
  LIBC_INLINE_ASM("fsd fs11, %0\n\t" : : "m"(buf->__fpregs[11]) :);
#elif defined(__riscv_float_abi_single)
#error "setjmp implementation not available for the target architecture."
#endif

  return 0;
}

} // namespace __llvm_libc
