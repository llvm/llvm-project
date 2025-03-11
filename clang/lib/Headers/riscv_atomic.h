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

#ifdef __riscv_zawrs
#define __riscv_wrs_nto __builtin_riscv_wrs_nto
#define __riscv_wrs_sto __builtin_riscv_wrs_sto
#endif

#endif
