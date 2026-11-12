#include "../assembly.h"

.arch armv8-a+sme2

// For Apple platforms at the moment, we just call abort() directly
// after stopping SM mode unconditionally.
.p2align 2
DEFINE_COMPILERRT_PRIVATE_FUNCTION(do_abort)
.cfi_startproc
	.variant_pcs	SYMBOL_NAME(do_abort)
	stp	x29, x30, [sp, #-32]!
  .cfi_def_cfa_offset 32
  .cfi_offset w30, -24
  .cfi_offset w29, -32
	smstop sm
	bl	SYMBOL_NAME(abort)
.cfi_endproc
END_COMPILERRT_FUNCTION(do_abort)

DEFINE_COMPILERRT_FUNCTION(__arm_tpidr2_save)
  // If TPIDR2_EL0 is null, the subroutine does nothing.
  mrs x16, TPIDR2_EL0
  cbz x16, 1f

  // If any of the reserved bytes in the first 16 bytes of the TPIDR2 block are
  // nonzero, the subroutine [..] aborts in some platform-defined manner.
  ldrh  w14, [x16, #10]
  cbnz  w14, 2f
  ldr w14, [x16, #12]
  cbnz  w14, 2f

  // If za_save_buffer is NULL, the subroutine does nothing.
  ldr x14, [x16]
  cbz x14, 1f

  // If num_za_save_slices is zero, the subroutine does nothing.
  ldrh  w14, [x16, #8]
  cbz x14, 1f

  mov x15, xzr
  ldr x16, [x16]
0:
  str za[w15,0], [x16]
  addsvl x16, x16, #1
  add x15, x15, #1
  cmp x14, x15
  b.ne  0b
1:
  ret
2:
  b  SYMBOL_NAME(do_abort)
END_COMPILERRT_FUNCTION(__arm_tpidr2_save)

.p2align 2
DEFINE_COMPILERRT_FUNCTION(__arm_za_disable)
.cfi_startproc
  // Otherwise, the subroutine behaves as if it did the following:
  // * Call __arm_tpidr2_save.
  stp x29, x30, [sp, #-16]!
  .cfi_def_cfa_offset 16
  mov x29, sp
  .cfi_def_cfa w29, 16
  .cfi_offset w30, -8
  .cfi_offset w29, -16
  bl  SYMBOL_NAME(__arm_tpidr2_save)

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

.p2align 2
DEFINE_COMPILERRT_FUNCTION(__arm_tpidr2_restore)
.cfi_startproc
  .variant_pcs	SYMBOL_NAME(__arm_tpidr2_restore)
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
  b  SYMBOL_NAME(do_abort)
.cfi_endproc
END_COMPILERRT_FUNCTION(__arm_tpidr2_restore)

.p2align 2
DEFINE_COMPILERRT_FUNCTION(__arm_sme_state)
	.variant_pcs	SYMBOL_NAME(__arm_sme_state)
  orr x0, x0, #0xC000000000000000
  mrs x16, SVCR
  bfxil x0, x16, #0, #2
  mrs x1, TPIDR2_EL0
  ret
END_COMPILERRT_FUNCTION(__arm_sme_state)
