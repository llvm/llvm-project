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

#define STORE_IMPL(insns, reg, val)                                            \
  LIBC_INLINE_ASM(#insns " " #reg ", %0\n\t" : : "m"(val) :)

#ifdef LIBC_TARGET_ARCH_IS_RISCV32
#define STORE(reg, val) STORE_IMPL(sw, reg, val)
#define STORE_FP(reg, val) STORE_IMPL(fsw, reg, val)
#else
#define STORE(reg, val) STORE_IMPL(sd, reg, val)
#define STORE_FP(reg, val) STORE_IMPL(fsd, reg, val)
#endif

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, setjmp, (__jmp_buf * buf)) {
  STORE(ra, buf->__pc);
  STORE(s0, buf->__regs[0]);
  STORE(s1, buf->__regs[1]);
  STORE(s2, buf->__regs[2]);
  STORE(s3, buf->__regs[3]);
  STORE(s4, buf->__regs[4]);
  STORE(s5, buf->__regs[5]);
  STORE(s6, buf->__regs[6]);
  STORE(s7, buf->__regs[7]);
  STORE(s8, buf->__regs[8]);
  STORE(s9, buf->__regs[9]);
  STORE(s10, buf->__regs[10]);
  STORE(s11, buf->__regs[11]);
  STORE(sp, buf->__sp);

#if __riscv_float_abi_double
  STORE_FP(fs0, buf->__fpregs[0]);
  STORE_FP(fs1, buf->__fpregs[1]);
  STORE_FP(fs2, buf->__fpregs[2]);
  STORE_FP(fs3, buf->__fpregs[3]);
  STORE_FP(fs4, buf->__fpregs[4]);
  STORE_FP(fs5, buf->__fpregs[5]);
  STORE_FP(fs6, buf->__fpregs[6]);
  STORE_FP(fs7, buf->__fpregs[7]);
  STORE_FP(fs8, buf->__fpregs[8]);
  STORE_FP(fs9, buf->__fpregs[9]);
  STORE_FP(fs10, buf->__fpregs[10]);
  STORE_FP(fs11, buf->__fpregs[11]);
#elif defined(__riscv_float_abi_single)
#error "setjmp implementation not available for the target architecture."
#endif

  return 0;
}

} // namespace LIBC_NAMESPACE
