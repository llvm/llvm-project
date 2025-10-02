//===----- riscv_mips.h - RISC-V MIPS Intrinsic definitions
//----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __RISCV_MIPS_H
#define __RISCV_MIPS_H

#if !defined(__riscv)
#error "This header is only meant to be used on riscv architecture"
#endif

#define __DEFAULT_FN_ATTRS                                                     \
  __attribute__((__always_inline__, __nodebug__, __target__("xmipsexectl")))

static __inline__ void __DEFAULT_FN_ATTRS __mips_pause() {
  __builtin_riscv_mips_pause();
}

static __inline__ void __DEFAULT_FN_ATTRS __mips_ehb() {
  __builtin_riscv_mips_ehb();
}

static __inline__ void __DEFAULT_FN_ATTRS __mips_ihb() {
  __builtin_riscv_mips_ihb();
}

#undef __DEFAULT_FN_ATTRS

#endif
