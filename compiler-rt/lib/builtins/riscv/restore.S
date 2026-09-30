//===-- restore.S - restore up to 12 callee-save registers ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Multiple entry points depending on number of registers to restore
//
//===----------------------------------------------------------------------===//

// All of the entry points are in the same section since we rely on many of
// them falling through into each other and don't want the linker to
// accidentally split them up, garbage collect, or reorder them.
//
// For the conventional ABIs, entry points are grouped up into 2s for rv64 and
// 4s for rv32 since this is the minimum grouping which will maintain the
// required 16-byte stack alignment.
//
// For the ilp32e/lp64e abis, entry points are grouped into 1s, since this is
// the minimum grouping which will maintain the required 4-byte stack alignment.

#include "assembly.h"

TEXT_SECTION

// Like DEFINE_COMPILERRT_FUNCTION but without:
// - Section change (TEXT_SECTION)
// - Subsections via symbols (FILE_LEVEL_DIRECTIVE)
//
// This is needed because `__riscv_restore_<N>` could fallthrough to a later
// `__riscv_restore_<N>` tail.
#define DEFINE_COMPILERRT_OVELAPPING_FUNCTION(name) \
  DEFINE_CODE_STATE \
  .globl FUNC_SYMBOL(SYMBOL_NAME(name)) SEPARATOR \
  SYMBOL_IS_FUNC(SYMBOL_NAME(name)) SEPARATOR \
  HIDDEN(SYMBOL_NAME(name)) SEPARATOR \
  DECLARE_FUNC_ENCODING \
  FUNC_SYMBOL(SYMBOL_NAME(name)):

#if __riscv_xlen == 32

#ifndef __riscv_abi_rve

DEFINE_COMPILERRT_OVELAPPING_FUNCTION(__riscv_restore_12)
  lw      s11, 12(sp)
  addi    sp, sp, 16
  // fallthrough into __riscv_restore_11

DEFINE_COMPILERRT_OVELAPPING_FUNCTION(__riscv_restore_11)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__riscv_restore_11, __riscv_restore_10)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__riscv_restore_11, __riscv_restore_9)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__riscv_restore_11, __riscv_restore_8)
  lw      s10, 0(sp)
  lw      s9,  4(sp)
  lw      s8,  8(sp)
  lw      s7,  12(sp)
  addi    sp, sp, 16
  // fallthrough into __riscv_restore_7

DEFINE_COMPILERRT_OVELAPPING_FUNCTION(__riscv_restore_7)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__riscv_restore_7, __riscv_restore_6)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__riscv_restore_7, __riscv_restore_5)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__riscv_restore_7, __riscv_restore_4)
  lw      s6,  0(sp)
  lw      s5,  4(sp)
  lw      s4,  8(sp)
  lw      s3,  12(sp)
  addi    sp, sp, 16
  // fallthrough into __riscv_restore_3

DEFINE_COMPILERRT_OVELAPPING_FUNCTION(__riscv_restore_3)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__riscv_restore_3, __riscv_restore_2)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__riscv_restore_3, __riscv_restore_1)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__riscv_restore_3, __riscv_restore_0)
  lw      s2,  0(sp)
  lw      s1,  4(sp)
  lw      s0,  8(sp)
  lw      ra,  12(sp)
  addi    sp, sp, 16
  ret
END_COMPILERRT_FUNCTION(__riscv_restore_12)
END_COMPILERRT_FUNCTION(__riscv_restore_11)
END_COMPILERRT_FUNCTION(__riscv_restore_7)
END_COMPILERRT_FUNCTION(__riscv_restore_3)

#else

DEFINE_COMPILERRT_OVELAPPING_FUNCTION(__riscv_restore_2)
  lw      s1, 0(sp)
  addi    sp, sp, 4
  // fallthrough into __riscv_restore_1/0

DEFINE_COMPILERRT_OVELAPPING_FUNCTION(__riscv_restore_1)
  lw      s0, 0(sp)
  addi    sp, sp, 4
  // fallthrough into __riscv_restore_0

DEFINE_COMPILERRT_OVELAPPING_FUNCTION(__riscv_restore_0)
  lw      ra, 0(sp)
  addi    sp, sp, 4
  ret
END_COMPILERRT_FUNCTION(__riscv_restore_2)
END_COMPILERRT_FUNCTION(__riscv_restore_1)
END_COMPILERRT_FUNCTION(__riscv_restore_0)

#endif

#elif __riscv_xlen == 64

#ifndef __riscv_abi_rve

DEFINE_COMPILERRT_OVELAPPING_FUNCTION(__riscv_restore_12)
  ld      s11, 8(sp)
  addi    sp, sp, 16
  // fallthrough into __riscv_restore_11/10

DEFINE_COMPILERRT_OVELAPPING_FUNCTION(__riscv_restore_11)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__riscv_restore_11, __riscv_restore_10)
  ld      s10, 0(sp)
  ld      s9,  8(sp)
  addi    sp, sp, 16
  // fallthrough into __riscv_restore_9/8

DEFINE_COMPILERRT_OVELAPPING_FUNCTION(__riscv_restore_9)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__riscv_restore_9, __riscv_restore_8)
  ld      s8,  0(sp)
  ld      s7,  8(sp)
  addi    sp, sp, 16
  // fallthrough into __riscv_restore_7/6

DEFINE_COMPILERRT_OVELAPPING_FUNCTION(__riscv_restore_7)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__riscv_restore_7, __riscv_restore_6)
  ld      s6,  0(sp)
  ld      s5,  8(sp)
  addi    sp, sp, 16
  // fallthrough into __riscv_restore_5/4

DEFINE_COMPILERRT_OVELAPPING_FUNCTION(__riscv_restore_5)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__riscv_restore_5, __riscv_restore_4)
  ld      s4,  0(sp)
  ld      s3,  8(sp)
  addi    sp, sp, 16
  // fallthrough into __riscv_restore_3/2

DEFINE_COMPILERRT_OVELAPPING_FUNCTION(__riscv_restore_3)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__riscv_restore_3, __riscv_restore_2)
  ld      s2,  0(sp)
  ld      s1,  8(sp)
  addi    sp, sp, 16
  // fallthrough into __riscv_restore_1/0

DEFINE_COMPILERRT_OVELAPPING_FUNCTION(__riscv_restore_1)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__riscv_restore_1, __riscv_restore_0)
  ld      s0,  0(sp)
  ld      ra,  8(sp)
  addi    sp, sp, 16
  ret
END_COMPILERRT_FUNCTION(__riscv_restore_12)
END_COMPILERRT_FUNCTION(__riscv_restore_11)
END_COMPILERRT_FUNCTION(__riscv_restore_9)
END_COMPILERRT_FUNCTION(__riscv_restore_7)
END_COMPILERRT_FUNCTION(__riscv_restore_5)
END_COMPILERRT_FUNCTION(__riscv_restore_3)
END_COMPILERRT_FUNCTION(__riscv_restore_1)

#else

DEFINE_COMPILERRT_OVELAPPING_FUNCTION(__riscv_restore_2)
  ld      s1, 0(sp)
  addi    sp, sp, 8
  // fallthrough into __riscv_restore_1/0

DEFINE_COMPILERRT_OVELAPPING_FUNCTION(__riscv_restore_1)
  ld      s0, 0(sp)
  addi    sp, sp, 8
  // fallthrough into __riscv_restore_0

DEFINE_COMPILERRT_OVELAPPING_FUNCTION(__riscv_restore_0)
  ld      ra, 0(sp)
  addi    sp, sp, 8
  ret

#endif

#else
# error "xlen must be 32 or 64 for save-restore implementation
#endif
