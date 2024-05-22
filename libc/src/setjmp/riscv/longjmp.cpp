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

#if !defined(LIBC_TARGET_ARCH_IS_ANY_RISCV)
#error "Invalid file include"
#endif

#define LOAD_IMPL(insns, reg, val)                                             \
  LIBC_INLINE_ASM(#insns " " #reg ", %0\n\t" : : "m"(val) :)

#ifdef LIBC_TARGET_ARCH_IS_RISCV32
#define LOAD(reg, val) LOAD_IMPL(lw, reg, val)
#define LOAD_FP(reg, val) LOAD_IMPL(flw, reg, val)
#else
#define LOAD(reg, val) LOAD_IMPL(ld, reg, val)
#define LOAD_FP(reg, val) LOAD_IMPL(fld, reg, val)
#endif

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(void, longjmp, (__jmp_buf * buf, int val)) {
  LOAD(ra, buf->__pc);
  LOAD(s0, buf->__regs[0]);
  LOAD(s1, buf->__regs[1]);
  LOAD(s2, buf->__regs[2]);
  LOAD(s3, buf->__regs[3]);
  LOAD(s4, buf->__regs[4]);
  LOAD(s5, buf->__regs[5]);
  LOAD(s6, buf->__regs[6]);
  LOAD(s7, buf->__regs[7]);
  LOAD(s8, buf->__regs[8]);
  LOAD(s9, buf->__regs[9]);
  LOAD(s10, buf->__regs[10]);
  LOAD(s11, buf->__regs[11]);
  LOAD(sp, buf->__sp);

#if __riscv_float_abi_double
  LOAD_FP(fs0, buf->__fpregs[0]);
  LOAD_FP(fs1, buf->__fpregs[1]);
  LOAD_FP(fs2, buf->__fpregs[2]);
  LOAD_FP(fs3, buf->__fpregs[3]);
  LOAD_FP(fs4, buf->__fpregs[4]);
  LOAD_FP(fs5, buf->__fpregs[5]);
  LOAD_FP(fs6, buf->__fpregs[6]);
  LOAD_FP(fs7, buf->__fpregs[7]);
  LOAD_FP(fs8, buf->__fpregs[8]);
  LOAD_FP(fs9, buf->__fpregs[9]);
  LOAD_FP(fs10, buf->__fpregs[10]);
  LOAD_FP(fs11, buf->__fpregs[11]);
#elif defined(__riscv_float_abi_single)
#error "longjmp implementation not available for the target architecture."
#endif

  val = val == 0 ? 1 : val;
  LIBC_INLINE_ASM("add a0, %0, zero\n\t" : : "r"(val) :);
}

} // namespace LIBC_NAMESPACE
