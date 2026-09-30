//===-- save.S - save up to 12 callee-saved registers ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Multiple entry points depending on number of registers to save
//
//===----------------------------------------------------------------------===//

// The entry points are grouped up into 2s for rv64 and 4s for rv32 since this
// is the minimum grouping which will maintain the required 16-byte stack
// alignment.
//
// We use lots of tail-sharing, which ends up looking like the functions
// overlap in the ELF metadata. This is expected - the alternative would be
// to separate the prologues into separate section and risk the
// `j LOCAL_LABEL(...)` targets being out-of-range at link time.

#include "assembly.h"

TEXT_SECTION

// Like DEFINE_COMPILERRT_FUNCTION but without:
// - Section change (TEXT_SECTION)
// - Subsections via symbols (FILE_LEVEL_DIRECTIVE)
//
// This is needed because `__riscv_save_<N>` could fallthrough to a later
// `__riscv_save_<N>` tail.
#define DEFINE_COMPILERRT_OVELAPPING_FUNCTION(name) \
  DEFINE_CODE_STATE \
  .globl FUNC_SYMBOL(SYMBOL_NAME(name)) SEPARATOR \
  SYMBOL_IS_FUNC(SYMBOL_NAME(name)) SEPARATOR \
  HIDDEN(SYMBOL_NAME(name)) SEPARATOR \
  DECLARE_FUNC_ENCODING \
  FUNC_SYMBOL(SYMBOL_NAME(name)):

#if __riscv_xlen == 32

#ifndef __riscv_abi_rve

DEFINE_COMPILERRT_OVELAPPING_FUNCTION(__riscv_save_12)
  addi   sp, sp, -64
  mv     t1, zero
  sw     s11, 12(sp)
  j      LOCAL_LABEL(riscv_save_11_8)

DEFINE_COMPILERRT_OVELAPPING_FUNCTION(__riscv_save_11)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__riscv_save_11, __riscv_save_10)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__riscv_save_11, __riscv_save_9)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__riscv_save_11, __riscv_save_8)
  addi   sp, sp, -64
  li     t1, 16
LOCAL_LABEL(riscv_save_11_8):
  sw     s10, 16(sp)
  sw     s9,  20(sp)
  sw     s8,  24(sp)
  sw     s7,  28(sp)
  j      LOCAL_LABEL(riscv_save_7_4)

DEFINE_COMPILERRT_OVELAPPING_FUNCTION(__riscv_save_7)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__riscv_save_7, __riscv_save_6)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__riscv_save_7, __riscv_save_5)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__riscv_save_7, __riscv_save_4)
  addi   sp, sp, -64
  li     t1, 32
LOCAL_LABEL(riscv_save_7_4):
  sw     s6, 32(sp)
  sw     s5, 36(sp)
  sw     s4, 40(sp)
  sw     s3, 44(sp)
  sw     s2, 48(sp)
  sw     s1, 52(sp)
  sw     s0, 56(sp)
  sw     ra, 60(sp)
  add    sp, sp, t1
  jr     t0
END_COMPILERRT_FUNCTION(__riscv_save_12)
END_COMPILERRT_FUNCTION(__riscv_save_11)
END_COMPILERRT_FUNCTION(__riscv_save_7)

// This function isn't overlapping, but we don't want to apply
// `.subsections_via_symbols` to the whole file because of the prior
// overlapping function.
DEFINE_COMPILERRT_OVELAPPING_FUNCTION(__riscv_save_3)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__riscv_save_3, __riscv_save_2)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__riscv_save_3, __riscv_save_1)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__riscv_save_3, __riscv_save_0)
  addi    sp, sp, -16
  sw      s2,  0(sp)
  sw      s1,  4(sp)
  sw      s0,  8(sp)
  sw      ra,  12(sp)
  jr      t0
END_COMPILERRT_FUNCTION(__riscv_save_3)

#else

DEFINE_COMPILERRT_FUNCTION(__riscv_save_2)
  addi    sp, sp, -12
  sw      s1,  0(sp)
  sw      s0,  4(sp)
  sw      ra,  8(sp)
  jr      t0
END_COMPILERRT_FUNCTION(__riscv_save_2)

DEFINE_COMPILERRT_FUNCTION(__riscv_save_1)
  addi    sp, sp, -8
  sw      s0,  0(sp)
  sw      ra,  4(sp)
  jr      t0
END_COMPILERRT_FUNCTION(__riscv_save_1)

DEFINE_COMPILERRT_FUNCTION(__riscv_save_0)
  addi    sp, sp, -4
  sw      ra,  0(sp)
  jr      t0
END_COMPILERRT_FUNCTION(__riscv_save_0)

#endif

#elif __riscv_xlen == 64

#ifndef __riscv_abi_rve

DEFINE_COMPILERRT_OVELAPPING_FUNCTION(__riscv_save_12)
  addi   sp, sp, -112
  mv     t1, zero
  sd     s11, 8(sp)
  j      LOCAL_LABEL(riscv_save_11_10)

DEFINE_COMPILERRT_OVELAPPING_FUNCTION(__riscv_save_11)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__riscv_save_11, __riscv_save_10)
  addi   sp, sp, -112
  li     t1, 16
LOCAL_LABEL(riscv_save_11_10):
  sd     s10, 16(sp)
  sd     s9,  24(sp)
  j      LOCAL_LABEL(riscv_save_9_8)

DEFINE_COMPILERRT_OVELAPPING_FUNCTION(__riscv_save_9)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__riscv_save_9, __riscv_save_8)
  addi   sp, sp, -112
  li     t1, 32
LOCAL_LABEL(riscv_save_9_8):
  sd     s8,  32(sp)
  sd     s7,  40(sp)
  j      LOCAL_LABEL(riscv_save_7_6)

DEFINE_COMPILERRT_OVELAPPING_FUNCTION(__riscv_save_7)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__riscv_save_7, __riscv_save_6)
  addi   sp, sp, -112
  li     t1, 48
LOCAL_LABEL(riscv_save_7_6):
  sd     s6,  48(sp)
  sd     s5,  56(sp)
  j      LOCAL_LABEL(riscv_save_5_4)

DEFINE_COMPILERRT_OVELAPPING_FUNCTION(__riscv_save_5)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__riscv_save_5, __riscv_save_4)
  addi   sp, sp, -112
  li     t1, 64
LOCAL_LABEL(riscv_save_5_4):
  sd     s4, 64(sp)
  sd     s3, 72(sp)
  j      LOCAL_LABEL(riscv_save_3_2)

DEFINE_COMPILERRT_OVELAPPING_FUNCTION(__riscv_save_3)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__riscv_save_3, __riscv_save_2)
  addi   sp, sp, -112
  li     t1, 80
LOCAL_LABEL(riscv_save_3_2):
  sd     s2, 80(sp)
  sd     s1, 88(sp)
  sd     s0, 96(sp)
  sd     ra, 104(sp)
  add    sp, sp, t1
  jr     t0
END_COMPILERRT_FUNCTION(__riscv_save_12)
END_COMPILERRT_FUNCTION(__riscv_save_11)
END_COMPILERRT_FUNCTION(__riscv_save_9)
END_COMPILERRT_FUNCTION(__riscv_save_7)
END_COMPILERRT_FUNCTION(__riscv_save_5)
END_COMPILERRT_FUNCTION(__riscv_save_3)

// This function isn't overlapping, but we don't want to apply
// `.subsections_via_symbols` to the whole file because of the prior
// overlapping function.
DEFINE_COMPILERRT_OVELAPPING_FUNCTION(__riscv_save_1)
DEFINE_COMPILERRT_FUNCTION_ALIAS(__riscv_save_1, __riscv_save_0)
__riscv_save_1:
__riscv_save_0:
  addi   sp, sp, -16
  sd     s0, 0(sp)
  sd     ra, 8(sp)
  jr     t0
END_COMPILERRT_FUNCTION(__riscv_save_1)

#else

DEFINE_COMPILERRT_FUNCTION(__riscv_save_2)
  addi    sp, sp, -24
  sw      s1, 0(sp)
  sw      s0, 8(sp)
  sw      ra, 16(sp)
  jr      t0
END_COMPILERRT_FUNCTION(__riscv_save_3)

DEFINE_COMPILERRT_FUNCTION(__riscv_save_1)
  addi    sp, sp, -16
  sw      s0, 0(sp)
  sw      ra, 8(sp)
  jr      t0
END_COMPILERRT_FUNCTION(__riscv_save_3)

DEFINE_COMPILERRT_FUNCTION(__riscv_save_0)
  addi    sp, sp, -8
  sw      ra,  0(sp)
  jr      t0
END_COMPILERRT_FUNCTION(__riscv_save_3)

#endif

#else
# error "xlen must be 32 or 64 for save-restore implementation
#endif
