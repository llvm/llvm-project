/*===---- riscv_atomic.h - RISC-V atomic intrinsics ------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __RISCV_ATOMIC_H
#define __RISCV_ATOMIC_H

#ifdef __riscv_zalrsc
enum {
  __RISCV_ORDERING_NONE = 0,
  __RISCV_ORDERING_AQ = 1,
  __RISCV_ORDERING_RL = 2,
  __RISCV_ORDERING_AQ_RL = 3
};

#define __riscv_lr_w __builtin_riscv_lr_w
#define __riscv_sc_w __builtin_riscv_sc_w

#if __riscv_xlen == 64
#define __riscv_lr_d __builtin_riscv_lr_d
#define __riscv_sc_d __builtin_riscv_sc_d
#endif

#endif

#ifdef __riscv_zawrs
#define __riscv_wrs_nto __builtin_riscv_wrs_nto
#define __riscv_wrs_sto __builtin_riscv_wrs_sto
#endif

#endif
