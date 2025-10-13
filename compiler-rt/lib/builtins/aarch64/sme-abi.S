// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This patch implements the support routines for the SME ABI,
// described here:
//  https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst#sme-support-routines

#include "../assembly.h"

.set FEAT_SVE_BIT, 30
.set FEAT_SME_BIT, 42
.set FEAT_SME2_BIT, 57
.set FEAT_SME2_MASK, 1 << 57
.set SVCR_PSTATE_SM_BIT, 0

#if !defined(__APPLE__)
#define CPU_FEATS_SYMBOL SYMBOL_NAME(__aarch64_cpu_features)
#define CPU_FEATS_SYMBOL_OFFSET :lo12:SYMBOL_NAME(__aarch64_cpu_features)
#else
// MachO requires @page/@pageoff directives because the global is defined
// in a different file. Otherwise this file may fail to build.
#define CPU_FEATS_SYMBOL SYMBOL_NAME(__aarch64_cpu_features)@page
#define CPU_FEATS_SYMBOL_OFFSET SYMBOL_NAME(__aarch64_cpu_features)@pageoff
#endif

.arch armv9-a+sme2

// Utility function which calls a system's abort() routine. Because the function
// is streaming-compatible it should disable streaming-SVE mode before calling
// abort(). Note that there is no need to preserve any state before the call,
// because the function does not return.
DEFINE_COMPILERRT_PRIVATE_FUNCTION(do_abort)
  .cfi_startproc
  .variant_pcs FUNC_SYMBOL(SYMBOL_NAME(do_abort))
  BTI_C
  stp  x29, x30, [sp, #-32]!
  cntd x0
  // Store VG to a stack location that we describe with .cfi_offset
  str x0, [sp, #16]
  .cfi_def_cfa_offset 32
  .cfi_offset w30, -24
  .cfi_offset w29, -32
  .cfi_offset 46, -16
  bl  FUNC_SYMBOL(SYMBOL_NAME(__arm_sme_state))
  tbz  x0, #0, 2f
1:
  smstop sm
2:
  // We can't make this into a tail-call because the unwinder would
  // need to restore the value of VG.
  bl  FUNC_SYMBOL(SYMBOL_NAME(abort))
  .cfi_endproc
END_COMPILERRT_FUNCTION(do_abort)

// __arm_sme_state fills the result registers based on a local
// that is set as part of the compiler-rt startup code.
//   __aarch64_has_sme_and_tpidr2_el0
DEFINE_COMPILERRT_FUNCTION(__arm_sme_state)
  .variant_pcs __arm_sme_state
  BTI_C
  mov x0, xzr
  mov x1, xzr

  adrp x16, CPU_FEATS_SYMBOL
  ldr x16, [x16, CPU_FEATS_SYMBOL_OFFSET]
  tbz x16, #FEAT_SME_BIT, 1f
0:
  orr x0, x0, #0xC000000000000000
  mrs x16, SVCR
  bfxil x0, x16, #0, #2
  mrs x1, TPIDR2_EL0
1:
  ret
END_COMPILERRT_FUNCTION(__arm_sme_state)

DEFINE_COMPILERRT_FUNCTION(__arm_tpidr2_restore)
  .variant_pcs __arm_tpidr2_restore
  BTI_C
  // If TPIDR2_EL0 is nonnull, the subroutine aborts in some platform-specific
  // manner.
  mrs x14, TPIDR2_EL0
  cbnz  x14, 2f

  // If any of the reserved bytes in the first 16 bytes of BLK are nonzero,
  // the subroutine [..] aborts in some platform-defined manner.
  ldrh  w14, [x0, #10]
  cbnz  w14, 2f
  ldr w14, [x0, #12]
  cbnz  w14, 2f

  // If BLK.za_save_buffer is NULL, the subroutine does nothing.
  ldr x16, [x0]
  cbz x16, 1f

  // If BLK.num_za_save_slices is zero, the subroutine does nothing.
  ldrh  w14, [x0, #8]
  cbz x14, 1f

  mov x15, xzr
0:
  ldr za[w15,0], [x16]
  addsvl x16, x16, #1
  add x15, x15, #1
  cmp x14, x15
  b.ne  0b
1:
  ret
2:
  b  FUNC_SYMBOL(SYMBOL_NAME(do_abort))
END_COMPILERRT_FUNCTION(__arm_tpidr2_restore)

DEFINE_COMPILERRT_FUNCTION(__arm_tpidr2_save)
  .variant_pcs __arm_tpidr2_save
  BTI_C
  // If the current thread does not have access to TPIDR2_EL0, the subroutine
  // does nothing.
  adrp x14, CPU_FEATS_SYMBOL
  ldr x14, [x14, CPU_FEATS_SYMBOL_OFFSET]
  tbz x14, #FEAT_SME_BIT, 1f

  // If TPIDR2_EL0 is null, the subroutine does nothing.
  mrs x16, TPIDR2_EL0
  cbz x16, 1f

  // If any of the reserved bytes in the first 16 bytes of the TPIDR2 block are
  // nonzero, the subroutine [..] aborts in some platform-defined manner.
  ldrh  w14, [x16, #10]
  cbnz  w14, 2f
  ldr w14, [x16, #12]
  cbnz  w14, 2f

  // If num_za_save_slices is zero, the subroutine does nothing.
  ldrh  w14, [x16, #8]
  cbz x14, 1f

  // If za_save_buffer is NULL, the subroutine does nothing.
  ldr x16, [x16]
  cbz x16, 1f

  mov x15, xzr
0:
  str za[w15,0], [x16]
  addsvl x16, x16, #1
  add x15, x15, #1
  cmp x14, x15
  b.ne  0b
1:
  ret
2:
  b  FUNC_SYMBOL(SYMBOL_NAME(do_abort))
END_COMPILERRT_FUNCTION(__arm_tpidr2_save)

DEFINE_COMPILERRT_FUNCTION(__arm_za_disable)
  .cfi_startproc
  .variant_pcs __arm_za_disable
  BTI_C
  // If the current thread does not have access to SME, the subroutine does
  // nothing.
  adrp x14, CPU_FEATS_SYMBOL
  ldr x14, [x14, CPU_FEATS_SYMBOL_OFFSET]
  tbz x14, #FEAT_SME_BIT, 0f

  // Otherwise, the subroutine behaves as if it did the following:
  // * Call __arm_tpidr2_save.
  stp x29, x30, [sp, #-16]!
  .cfi_def_cfa_offset 16
  mov x29, sp
  .cfi_def_cfa w29, 16
  .cfi_offset w30, -8
  .cfi_offset w29, -16
  bl  FUNC_SYMBOL(SYMBOL_NAME(__arm_tpidr2_save))

  // * Set TPIDR2_EL0 to null.
  msr TPIDR2_EL0, xzr

  // * Set PSTATE.ZA to 0.
  smstop za

  .cfi_def_cfa wsp, 16
  ldp x29, x30, [sp], #16
  .cfi_def_cfa_offset 0
  .cfi_restore w30
  .cfi_restore w29
0:
  ret
  .cfi_endproc
END_COMPILERRT_FUNCTION(__arm_za_disable)

DEFINE_COMPILERRT_FUNCTION(__arm_get_current_vg)
  .variant_pcs __arm_get_current_vg
  BTI_C

  adrp    x17, CPU_FEATS_SYMBOL
  ldr     x17, [x17, CPU_FEATS_SYMBOL_OFFSET]
  tbnz    w17, #FEAT_SVE_BIT, 1f
  tbz     x17, #FEAT_SME_BIT, 2f
0:
  mrs     x17, SVCR
  tbz     x17, #SVCR_PSTATE_SM_BIT, 2f
1:
  cntd    x0
  ret
2:
  mov     x0, xzr
  ret
END_COMPILERRT_FUNCTION(__arm_get_current_vg)

// The diagram below describes the layout used in the following routines:
// * __arm_sme_state_size
// * __arm_sme_save
// * __arm_sme_restore
//
// +---------------------------------+
// |             ...                 |
// |           ZA buffer             |
// |             ...                 |
// +---------------------------------+ <- @96
// |         ZT0 contents            |
// +---------------------------------+ <- @32
// | byte 15-10: zero (reserved)     |
// | byte   9-8: num_za_save_slices  |           TPIDR2 block
// | byte   7-0: za_save_buffer      |
// +---------------------------------+ <- @16
// | bit  127-1: zero (reserved)     |           Internal state for __arm_sme_save/restore
// | bit      0: VALID               |
// +---------------------------------+ <- @0

DEFINE_COMPILERRT_FUNCTION(__arm_sme_state_size)
  .variant_pcs __arm_sme_state_size
  BTI_C

  // Test if SME is available and ZA state is 'active'.
  adrp    x17, CPU_FEATS_SYMBOL
  ldr     x17, [x17, CPU_FEATS_SYMBOL_OFFSET]
  tbz     x17, #FEAT_SME_BIT, 0f
  mrs     x16, SVCR
  tbz     x16, #1, 0f
  mrs     x16, TPIDR2_EL0
  cbnz    x16, 0f

  // Size = HAS_FEAT_SME2 ? 96 : 32
  tst     x17, #FEAT_SME2_MASK
  mov     w17, #32
  mov     w16, #96
  csel    x16, x17, x16, eq

  // Size = Size + (SVLB * SVLB)
  rdsvl   x17, #1
  madd    x0, x17, x17, x16
  ret

0:
  // Default case, 16 bytes is minimum (to encode VALID bit, multiple of 16 bytes)
  mov w0, #16
  ret
END_COMPILERRT_FUNCTION(__arm_sme_state_size)

DEFINE_COMPILERRT_FUNCTION(__arm_sme_save)
  .variant_pcs __arm_sme_save
  BTI_C

  // If PTR is not 16-byte aligned, abort.
  tst     x0, #0xF
  b.ne    3f

  // Clear internal state bits
  stp     xzr, xzr, [x0]

  // If SME is not available, PSTATE.ZA = 0 or TPIDR2_EL0 != 0, return.
  adrp    x17, CPU_FEATS_SYMBOL
  ldr     x17, [x17, CPU_FEATS_SYMBOL_OFFSET]
  tbz     x17, #FEAT_SME_BIT, 2f
  mrs     x16, SVCR
  tbz     x16, #1, 2f
  mrs     x16, TPIDR2_EL0
  cbnz    x16, 2f

  # ZA or ZT0 need saving, we can now set internal VALID bit to 1
  mov     w16, #1
  str     x16, [x0]

  add     x16, x0, #32
  tbz     x17, #FEAT_SME2_BIT, 1f

  // Store ZT0
  str     zt0, [x16]
  add     x16, x16, #64

1:
  // Set up lazy-save (x16 = pointer to buffer)
  rdsvl   x17, #1
  str     x16, [x0, #16]!
  strh    w17, [x0, #8]
  strh    wzr, [x0, #10]
  str     wzr, [x0, #12]
  msr     TPIDR2_EL0, x0

2:
  // Do nothing
  ret

3:
  b       FUNC_SYMBOL(SYMBOL_NAME(do_abort))
END_COMPILERRT_FUNCTION(__arm_sme_save)

DEFINE_COMPILERRT_FUNCTION(__arm_sme_restore)
  .cfi_startproc
  .variant_pcs __arm_sme_restore
  BTI_C

  stp     x29, x30, [sp, #-16]!
  .cfi_def_cfa_offset 16
  mov     x29, sp
  .cfi_def_cfa w29, 16
  .cfi_offset w30, -8
  .cfi_offset w29, -16

  // If PTR is not 16-byte aligned, abort.
  tst     x0, #0xF
  b.ne    3f

  // If the VALID bit is 0, return early.
  ldr     x16, [x0]
  cbz     x16, 2f

  // If SME is not available, abort.
  adrp    x17, CPU_FEATS_SYMBOL
  ldr     x17, [x17, CPU_FEATS_SYMBOL_OFFSET]
  tbz     x17, #FEAT_SME_BIT, 3f

  // If TPIDR2_EL0 != nullptr, no lazy-save was committed, try to reload zt0.
  mrs     x16, TPIDR2_EL0
  cbnz    x16, 1f

  // If TPIDR2_EL0 == nullptr and PSTATE.ZA = 1 (<=> ZA state is 'active'),
  // abort.
  mrs     x16, SVCR
  tbnz    x16, #1, 3f

  // Restore za.
  smstart za
  add     x0, x0, #16
  bl      __arm_tpidr2_restore
  sub     x0, x0, #16

1:
  smstart za
  msr     TPIDR2_EL0, xzr

  // Check if zt0 needs restoring.
  tbz     x17, #FEAT_SME2_BIT, 2f

  // Restore zt0.
  add     x16, x0, #32
  ldr     zt0, [x16]

2:
  // Do nothing
  .cfi_def_cfa wsp, 16
  ldp     x29, x30, [sp], #16
  .cfi_def_cfa_offset 0
  .cfi_restore w30
  .cfi_restore w29
  ret

3:
  b       FUNC_SYMBOL(SYMBOL_NAME(do_abort))
  .cfi_endproc
END_COMPILERRT_FUNCTION(__arm_sme_restore)

NO_EXEC_STACK_DIRECTIVE

// GNU property note for BTI, PAC, and GCS
GNU_PROPERTY_BTI_PAC_GCS
