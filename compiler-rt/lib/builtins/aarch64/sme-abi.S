// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This patch implements the support routines for the SME ABI,
// described here:
//  https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst#sme-support-routines

#include "../assembly.h"

#ifdef HAS_ASM_SME
#define ARCH armv9-a+sme
#define SMSTOP_SM smstop sm
#define SMSTOP_ZA smstop za
#define REG_TPIDR2_EL0 TPIDR2_EL0
#define REG_SVCR SVCR
#define ADDSVL_X16_X16_1 addsvl x16, x16, #1
#define LDR_ZA_W15_0_X16 ldr za[w15,0], [x16]
#define STR_ZA_W15_0_X16 str za[w15,0], [x16]
#define CNTD_X0 cntd x0
#define CFI_OFFSET_VG_MINUS_16 .cfi_offset vg, -16
#else
#define ARCH armv8-a
#define SMSTOP_SM .inst 0xd503427f
#define SMSTOP_ZA .inst 0xd503447f
#define REG_TPIDR2_EL0 S3_3_C13_C0_5
#define REG_SVCR S3_3_C4_C2_2
#define ADDSVL_X16_X16_1 .inst 0x04305830
#define LDR_ZA_W15_0_X16 .inst 0xe1006200
#define STR_ZA_W15_0_X16 .inst 0xe1206200
#define CNTD_X0 .inst 0x04e0e3e0
#define CFI_OFFSET_VG_MINUS_16 .cfi_escape 0x10, 0x2e, 0x03, 0x11, 0x70, 0x22 // $vg  @ cfa - 16
#endif

.arch ARCH

// Utility function which calls a system's abort() routine. Because the function
// is streaming-compatible it should disable streaming-SVE mode before calling
// abort(). Note that there is no need to preserve any state before the call,
// because the function does not return.
DEFINE_COMPILERRT_PRIVATE_FUNCTION(do_abort)
.cfi_startproc
	.variant_pcs	SYMBOL_NAME(do_abort)
	stp	x29, x30, [sp, #-32]!
  CNTD_X0
  // Store VG to a stack location that we describe with .cfi_offset
  str x0, [sp, #16]
  .cfi_def_cfa_offset 32
  .cfi_offset w30, -24
  .cfi_offset w29, -32
  CFI_OFFSET_VG_MINUS_16
	bl	__arm_sme_state
	tbz	x0, #0, 2f
1:
	SMSTOP_SM
2:
  // We can't make this into a tail-call because the unwinder would
  // need to restore the value of VG.
	bl	SYMBOL_NAME(abort)
.cfi_endproc
END_COMPILERRT_FUNCTION(do_abort)

// __arm_sme_state fills the result registers based on a local
// that is set as part of the compiler-rt startup code.
//   __aarch64_has_sme_and_tpidr2_el0
DEFINE_COMPILERRT_OUTLINE_FUNCTION_UNMANGLED(__arm_sme_state)
	.variant_pcs	__arm_sme_state
  mov x0, xzr
  mov x1, xzr

  adrp  x16, SYMBOL_NAME(__aarch64_has_sme_and_tpidr2_el0)
  ldrb w16, [x16, :lo12:SYMBOL_NAME(__aarch64_has_sme_and_tpidr2_el0)]
  cbz w16, 1f
0:
  orr x0, x0, #0xC000000000000000
  mrs x16, REG_SVCR
  bfxil x0, x16, #0, #2
  mrs x1, REG_TPIDR2_EL0
1:
  ret
END_COMPILERRT_OUTLINE_FUNCTION(__arm_sme_state)

DEFINE_COMPILERRT_OUTLINE_FUNCTION_UNMANGLED(__arm_tpidr2_restore)
	.variant_pcs	__arm_tpidr2_restore
  // If TPIDR2_EL0 is nonnull, the subroutine aborts in some platform-specific
  // manner.
  mrs x14, REG_TPIDR2_EL0
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
  LDR_ZA_W15_0_X16
  ADDSVL_X16_X16_1
  add x15, x15, #1
  cmp x14, x15
  b.ne  0b
1:
  ret
2:
  b  SYMBOL_NAME(do_abort)
END_COMPILERRT_OUTLINE_FUNCTION(__arm_tpidr2_restore)

DEFINE_COMPILERRT_OUTLINE_FUNCTION_UNMANGLED(__arm_tpidr2_save)
  // If the current thread does not have access to TPIDR2_EL0, the subroutine
  // does nothing.
  adrp  x14, SYMBOL_NAME(__aarch64_has_sme_and_tpidr2_el0)
  ldrb w14, [x14, :lo12:SYMBOL_NAME(__aarch64_has_sme_and_tpidr2_el0)]
  cbz w14, 1f

  // If TPIDR2_EL0 is null, the subroutine does nothing.
  mrs x16, REG_TPIDR2_EL0
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
  STR_ZA_W15_0_X16
  ADDSVL_X16_X16_1
  add x15, x15, #1
  cmp x14, x15
  b.ne  0b
1:
  ret
2:
  b  SYMBOL_NAME(do_abort)
END_COMPILERRT_OUTLINE_FUNCTION(__arm_tpidr2_save)

DEFINE_COMPILERRT_OUTLINE_FUNCTION_UNMANGLED(__arm_za_disable)
  // If the current thread does not have access to SME, the subroutine does
  // nothing.
  adrp  x14, SYMBOL_NAME(__aarch64_has_sme_and_tpidr2_el0)
  ldrb w14, [x14, :lo12:SYMBOL_NAME(__aarch64_has_sme_and_tpidr2_el0)]
  cbz w14, 0f

  // Otherwise, the subroutine behaves as if it did the following:
  // * Call __arm_tpidr2_save.
  stp x29, x30, [sp, #-16]!
  .cfi_def_cfa_offset 16
  mov x29, sp
  .cfi_def_cfa w29, 16
  .cfi_offset w30, -8
  .cfi_offset w29, -16
  bl  __arm_tpidr2_save

  // * Set TPIDR2_EL0 to null.
  msr REG_TPIDR2_EL0, xzr

  // * Set PSTATE.ZA to 0.
  SMSTOP_ZA

  .cfi_def_cfa wsp, 16
  ldp x29, x30, [sp], #16
  .cfi_def_cfa_offset 0
  .cfi_restore w30
  .cfi_restore w29
0:
  ret
END_COMPILERRT_OUTLINE_FUNCTION(__arm_za_disable)
