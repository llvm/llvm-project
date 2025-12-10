// RUN: %clang %cflags -march=armv8.3-a %s -o %t.exe -Wl,--emit-relocs
// RUN: llvm-bolt-binary-analysis --scanners=pauth %t.exe 2>&1 | FileCheck %s

        .text

        .globl  raise_error
        .type   raise_error,@function
raise_error:
        ret
        .size raise_error, .-raise_error

        .globl  resign_no_check
        .type   resign_no_check,@function
resign_no_check:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function resign_no_check, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacia x0, x2
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        autib   x0, x1
        pacia   x0, x2
        ret
        .size resign_no_check, .-resign_no_check

// Test "xpac" check method.

        .globl  resign_xpaci_good
        .type   resign_xpaci_good,@function
resign_xpaci_good:
// CHECK-NOT: resign_xpaci_good
        autib   x0, x1
        mov     x16, x0
        xpaci   x16
        cmp     x0, x16
        b.eq    1f
        brk     0xc471
1:
        pacia   x0, x2
        ret
        .size resign_xpaci_good, .-resign_xpaci_good

        .globl  resign_xpacd_good
        .type   resign_xpacd_good,@function
resign_xpacd_good:
// CHECK-NOT: resign_xpacd_good
        autdb   x0, x1
        mov     x16, x0
        xpacd   x16
        cmp     x0, x16
        b.eq    1f
        brk     0xc473
1:
        pacda   x0, x2
        ret
        .size resign_xpacd_good, .-resign_xpacd_good

        .globl  resign_xpaci_wrong_error
        .type   resign_xpaci_wrong_error,@function
resign_xpaci_wrong_error:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function resign_xpaci_wrong_error, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacia   x0, x2
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      bl      raise_error
        paciasp
        stp     x29, x30, [sp, #-16]!

        autib   x0, x1
        mov     x16, x0
        xpaci   x16
        cmp     x0, x16
        b.eq    1f
        bl      raise_error  // should trigger breakpoint instead
1:
        pacia   x0, x2

        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size resign_xpaci_wrong_error, .-resign_xpaci_wrong_error

        .globl  resign_xpaci_missing_brk
        .type   resign_xpaci_missing_brk,@function
resign_xpaci_missing_brk:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function resign_xpaci_missing_brk, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacia   x0, x2
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        autib   x0, x1
        mov     x16, x0
        xpaci   x16
        cmp     x0, x16
        b.eq    1f
1:
        pacia   x0, x2
        ret
        .size resign_xpaci_missing_brk, .-resign_xpaci_missing_brk

        .globl  resign_xpaci_missing_branch_and_brk
        .type   resign_xpaci_missing_branch_and_brk,@function
resign_xpaci_missing_branch_and_brk:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function resign_xpaci_missing_branch_and_brk, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacia   x0, x2
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        autib   x0, x1
        mov     x16, x0
        xpaci   x16
        cmp     x0, x16
        pacia   x0, x2
        ret
        .size resign_xpaci_missing_branch_and_brk, .-resign_xpaci_missing_branch_and_brk

        .globl  resign_xpaci_unrelated_auth_and_check
        .type   resign_xpaci_unrelated_auth_and_check,@function
resign_xpaci_unrelated_auth_and_check:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function resign_xpaci_unrelated_auth_and_check, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacia   x10, x2
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        autib   x10, x1  // made x10 safe-to-dereference
        mov     x16, x0  // start of checker sequence for x0
        xpaci   x16
        cmp     x0, x16
        b.eq    1f
        brk     0xc471
1:
        pacia   x10, x2
        ret
        .size resign_xpaci_unrelated_auth_and_check, .-resign_xpaci_unrelated_auth_and_check

// There are lots of operands to check in the pattern - let's at the very least
// check that each of the three instructions (mov, xpac, cmp) undergoes *some*
// matching. Pay a bit more attention to those instructions and their operands
// that can be obviously replaced without crashing at run-time and making the
// check obviously weaker.
        .globl  resign_xpaci_wrong_pattern_1
        .type   resign_xpaci_wrong_pattern_1,@function
resign_xpaci_wrong_pattern_1:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function resign_xpaci_wrong_pattern_1, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacia   x0, x2
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        autib   x0, x1
        mov     x16, x10  // x10 instead of x0
        xpaci   x16
        cmp     x0, x16
        b.eq    1f
        brk     0xc471
1:
        pacia   x0, x2
        ret
        .size resign_xpaci_wrong_pattern_1, .-resign_xpaci_wrong_pattern_1

        .globl  resign_xpaci_wrong_pattern_2
        .type   resign_xpaci_wrong_pattern_2,@function
resign_xpaci_wrong_pattern_2:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function resign_xpaci_wrong_pattern_2, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacia   x0, x2
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      xpaci   x0
        autib   x0, x1
        mov     x16, x0
        xpaci   x0        // x0 instead of x16
        cmp     x0, x16
        b.eq    1f
        brk     0xc471
1:
        pacia   x0, x2
        ret
        .size resign_xpaci_wrong_pattern_2, .-resign_xpaci_wrong_pattern_2

        .globl  resign_xpaci_wrong_pattern_3
        .type   resign_xpaci_wrong_pattern_3,@function
resign_xpaci_wrong_pattern_3:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function resign_xpaci_wrong_pattern_3, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacia   x0, x2
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        autib   x0, x1
        mov     x16, x0
        xpaci   x16
        cmp     x16, x16  // x16 instead of x0
        b.eq    1f
        brk     0xc471
1:
        pacia   x0, x2
        ret
        .size resign_xpaci_wrong_pattern_3, .-resign_xpaci_wrong_pattern_3

        .globl  resign_xpaci_wrong_pattern_4
        .type   resign_xpaci_wrong_pattern_4,@function
resign_xpaci_wrong_pattern_4:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function resign_xpaci_wrong_pattern_4, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacia   x0, x2
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        autib   x0, x1
        mov     x16, x0
        xpaci   x16
        cmp     x0, x0    // x0 instead of x16
        b.eq    1f
        brk     0xc471
1:
        pacia   x0, x2
        ret
        .size resign_xpaci_wrong_pattern_4, .-resign_xpaci_wrong_pattern_4

        .globl  resign_xpaci_wrong_pattern_5
        .type   resign_xpaci_wrong_pattern_5,@function
resign_xpaci_wrong_pattern_5:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function resign_xpaci_wrong_pattern_5, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacia   x0, x2
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        autib   x0, x1
        mov     x16, x0
        mov     x16, x16  // replace xpaci with a no-op instruction
        cmp     x0, x16
        b.eq    1f
        brk     0xc471
1:
        pacia   x0, x2
        ret
        .size resign_xpaci_wrong_pattern_5, .-resign_xpaci_wrong_pattern_5

// Test "xpac-hint" check method.

        .globl  resign_xpaclri_good
        .type   resign_xpaclri_good,@function
resign_xpaclri_good:
// CHECK-NOT: resign_xpaclri_good
        paciasp
        stp     x29, x30, [sp, #-16]!

        autib   x30, x1
        mov     x16, x30
        xpaclri
        cmp     x30, x16
        b.eq    1f
        brk     0xc471
1:
        pacia   x30, x2

        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size resign_xpaclri_good, .-resign_xpaclri_good

        .globl  xpaclri_check_keeps_lr_safe
        .type   xpaclri_check_keeps_lr_safe,@function
xpaclri_check_keeps_lr_safe:
// CHECK-NOT: xpaclri_check_keeps_lr_safe
        // LR is implicitly safe-to-dereference and trusted here
        mov     x16, x30
        xpaclri         // clobbers LR
        cmp     x30, x16
        b.eq    1f
        brk     0xc471    // marks LR as trusted and safe-to-dereference
1:
        ret             // not reporting non-protected return
        .size xpaclri_check_keeps_lr_safe, .-xpaclri_check_keeps_lr_safe

        .globl  xpaclri_check_requires_safe_lr
        .type   xpaclri_check_requires_safe_lr,@function
xpaclri_check_requires_safe_lr:
// CHECK-LABEL: GS-PAUTH: non-protected ret found in function xpaclri_check_requires_safe_lr, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      ret
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      xpaclri
        mov     x30, x0
        // LR is not safe-to-dereference here - check that xpac-hint checker
        // does not make LR safe-to-dereference, but only *keeps* this state.
        mov     x16, x30
        xpaclri
        cmp     x30, x16
        b.eq    1f
        brk     0xc471
1:
        ret
        .size xpaclri_check_requires_safe_lr, .-xpaclri_check_requires_safe_lr

        .globl  resign_xpaclri_wrong_reg
        .type   resign_xpaclri_wrong_reg,@function
resign_xpaclri_wrong_reg:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function resign_xpaclri_wrong_reg, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacia   x20, x2
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        paciasp

        autib   x20, x1   // consistently using x20 instead of x30
        mov     x16, x20
        xpaclri         // ... but xpaclri still operates on x30
        cmp     x20, x16
        b.eq    1f
        brk     0xc471
1:
        pacia   x20, x2

        autiasp
        ret
        .size resign_xpaclri_wrong_reg, .-resign_xpaclri_wrong_reg

// Test that pointer should be authenticated AND checked to be safe-to-sign.
// Checking alone is not enough.
        .globl  resign_checked_not_authenticated
        .type   resign_checked_not_authenticated,@function
resign_checked_not_authenticated:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function resign_checked_not_authenticated, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacia   x0, x2
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        mov     x16, x0
        xpaci   x16
        cmp     x0, x16
        b.eq    1f
        brk     0xc471
1:
        pacia   x0, x2
        ret
        .size resign_checked_not_authenticated, .-resign_checked_not_authenticated

// The particular register should be *first* written by an authentication
// instruction and *then* that new value should be checked.
// Such code pattern will probably crash at run-time anyway, but let's check
// "safe-to-dereference" -> "trusted" transition.
        .globl  resign_checked_before_authenticated
        .type   resign_checked_before_authenticated,@function
resign_checked_before_authenticated:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function resign_checked_before_authenticated, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacia   x0, x2
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        mov     x16, x0
        xpaci   x16
        cmp     x0, x16
        b.eq    1f
        brk     0xc471
1:
        autib   x0, x1
        pacia   x0, x2
        ret
        .size resign_checked_before_authenticated, .-resign_checked_before_authenticated

// Test "high-bits-notbi" check method.

        .globl  resign_high_bits_tbz_good
        .type   resign_high_bits_tbz_good,@function
resign_high_bits_tbz_good:
// CHECK-NOT: resign_high_bits_tbz_good
        autib   x0, x1
        eor     x16, x0, x0, lsl #1
        tbz     x16, #62, 1f
        brk     0xc471
1:
        pacia   x0, x2
        ret
        .size resign_high_bits_tbz_good, .-resign_high_bits_tbz_good

// Check BRK matching briefly - this logic is shared with the "xpac" sequence matcher.

        .globl  resign_high_bits_tbz_wrong_error
        .type   resign_high_bits_tbz_wrong_error,@function
resign_high_bits_tbz_wrong_error:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function resign_high_bits_tbz_wrong_error, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacia   x0, x2
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      bl      raise_error
        paciasp
        stp     x29, x30, [sp, #-16]!

        autib   x0, x1
        eor     x16, x0, x0, lsl #1
        tbz     x16, #62, 1f
        bl      raise_error    // should trigger breakpoint instead
1:
        pacia   x0, x2

        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size resign_high_bits_tbz_wrong_error, .-resign_high_bits_tbz_wrong_error

        .globl  resign_high_bits_tbz_wrong_bit
        .type   resign_high_bits_tbz_wrong_bit,@function
resign_high_bits_tbz_wrong_bit:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function resign_high_bits_tbz_wrong_bit, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacia   x0, x2
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        autib   x0, x1
        eor     x16, x0, x0, lsl #1
        tbz     x16, #63, 1f
        brk     0xc471
1:
        pacia   x0, x2
        ret
        .size resign_high_bits_tbz_wrong_bit, .-resign_high_bits_tbz_wrong_bit

        .globl  resign_high_bits_tbz_wrong_shift_amount
        .type   resign_high_bits_tbz_wrong_shift_amount,@function
resign_high_bits_tbz_wrong_shift_amount:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function resign_high_bits_tbz_wrong_shift_amount, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacia   x0, x2
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        autib   x0, x1
        eor     x16, x0, x0, lsl #2
        tbz     x16, #62, 1f
        brk     0xc471
1:
        pacia   x0, x2
        ret
        .size resign_high_bits_tbz_wrong_shift_amount, .-resign_high_bits_tbz_wrong_shift_amount

        .globl  resign_high_bits_tbz_wrong_shift_type
        .type   resign_high_bits_tbz_wrong_shift_type,@function
resign_high_bits_tbz_wrong_shift_type:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function resign_high_bits_tbz_wrong_shift_type, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacia   x0, x2
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        autib   x0, x1
        eor     x16, x0, x0, lsr #1
        tbz     x16, #62, 1f
        brk     0xc471
1:
        pacia   x0, x2
        ret
        .size resign_high_bits_tbz_wrong_shift_type, .-resign_high_bits_tbz_wrong_shift_type

        .globl  resign_high_bits_tbz_wrong_pattern_1
        .type   resign_high_bits_tbz_wrong_pattern_1,@function
resign_high_bits_tbz_wrong_pattern_1:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function resign_high_bits_tbz_wrong_pattern_1, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacia   x0, x2
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        autib   x0, x1
        eor     x16, x0, x0, lsl #1
        tbz     x17, #62, 1f
        brk     0xc471
1:
        pacia   x0, x2
        ret
        .size resign_high_bits_tbz_wrong_pattern_1, .-resign_high_bits_tbz_wrong_pattern_1

        .globl  resign_high_bits_tbz_wrong_pattern_2
        .type   resign_high_bits_tbz_wrong_pattern_2,@function
resign_high_bits_tbz_wrong_pattern_2:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function resign_high_bits_tbz_wrong_pattern_2, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacia   x0, x2
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        autib   x0, x1
        eor     x16, x10, x0, lsl #1
        tbz     x16, #62, 1f
        brk     0xc471
1:
        pacia   x0, x2
        ret
        .size resign_high_bits_tbz_wrong_pattern_2, .-resign_high_bits_tbz_wrong_pattern_2

        .globl  resign_high_bits_tbz_wrong_pattern_3
        .type   resign_high_bits_tbz_wrong_pattern_3,@function
resign_high_bits_tbz_wrong_pattern_3:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function resign_high_bits_tbz_wrong_pattern_3, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacia   x0, x2
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        autib   x0, x1
        eor     x16, x0, x10, lsl #1
        tbz     x16, #62, 1f
        brk     0xc471
1:
        pacia   x0, x2
        ret
        .size resign_high_bits_tbz_wrong_pattern_3, .-resign_high_bits_tbz_wrong_pattern_3

// Test checking by loading via the authenticated pointer.

        .globl  resign_load_good
        .type   resign_load_good,@function
resign_load_good:
// CHECK-NOT: resign_load_good
        autdb   x0, x1
        ldr     x3, [x0]
        pacda   x0, x2
        ret
        .size resign_load_good, .-resign_load_good

        .globl  resign_load_wreg_good
        .type   resign_load_wreg_good,@function
resign_load_wreg_good:
// CHECK-NOT: resign_load_wreg_good
        autdb   x0, x1
        ldr     w3, [x0]
        pacda   x0, x2
        ret
        .size resign_load_wreg_good, .-resign_load_wreg_good

        .globl  resign_load_byte_good
        .type   resign_load_byte_good,@function
resign_load_byte_good:
// CHECK-NOT: resign_load_byte_good
        autdb   x0, x1
        ldrb    w3, [x0]
        pacda   x0, x2
        ret
        .size resign_load_byte_good, .-resign_load_byte_good

        .globl  resign_load_pair_good
        .type   resign_load_pair_good,@function
resign_load_pair_good:
// CHECK-NOT: resign_load_pair_good
        autdb   x0, x1
        ldp     x3, x4, [x0]
        pacda   x0, x2
        ret
        .size resign_load_pair_good, .-resign_load_pair_good

        .globl  resign_load_imm_offset_good
        .type   resign_load_imm_offset_good,@function
resign_load_imm_offset_good:
// CHECK-NOT: resign_load_imm_offset_good
        autdb   x0, x1
        ldr     x3, [x0, #16]
        pacda   x0, x2
        ret
        .size resign_load_imm_offset_good, .-resign_load_imm_offset_good

        .globl  resign_load_preinc_good
        .type   resign_load_preinc_good,@function
resign_load_preinc_good:
// CHECK-NOT: resign_load_preinc_good
        autdb   x0, x1
        ldr     x3, [x0, #16]!
        pacda   x0, x2
        ret
        .size resign_load_preinc_good, .-resign_load_preinc_good

        .globl  resign_load_postinc_good
        .type   resign_load_postinc_good,@function
resign_load_postinc_good:
// CHECK-NOT: resign_load_postinc_good
        autdb   x0, x1
        ldr     x3, [x0], #16
        pacda   x0, x2
        ret
        .size resign_load_postinc_good, .-resign_load_postinc_good

        .globl  resign_load_pair_with_ptr_writeback_good
        .type   resign_load_pair_with_ptr_writeback_good,@function
resign_load_pair_with_ptr_writeback_good:
// CHECK-NOT: resign_load_pair_with_ptr_writeback_good
        autdb   x0, x1
        ldp     x3, x4, [x0, #16]!  // three output registers (incl. tied x0 register)
        pacda   x0, x2
        ret
        .size resign_load_pair_with_ptr_writeback_good, .-resign_load_pair_with_ptr_writeback_good

        .globl  resign_load_overwrite
        .type   resign_load_overwrite,@function
resign_load_overwrite:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function resign_load_overwrite, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacda   x0, x2
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      ldr     x0, [x0]
        autdb   x0, x1
        ldr     x0, [x0]
        pacda   x0, x2
        ret
        .size resign_load_overwrite, .-resign_load_overwrite

        .globl  resign_load_overwrite_out2
        .type   resign_load_overwrite_out2,@function
resign_load_overwrite_out2:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function resign_load_overwrite_out2, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacda   x0, x2
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      ldp     x10, x0, [x0]
        autdb   x0, x1
        ldp     x10, x0, [x0]
        pacda   x0, x2
        ret
        .size resign_load_overwrite_out2, .-resign_load_overwrite_out2

        .globl  resign_load_partial_overwrite
        .type   resign_load_partial_overwrite,@function
resign_load_partial_overwrite:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function resign_load_partial_overwrite, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacda   x0, x2
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      ldr     w0, [x0]
        autdb   x0, x1
        ldr     w0, [x0]
        pacda   x0, x2
        ret
        .size resign_load_partial_overwrite, .-resign_load_partial_overwrite

        .globl  resign_load_partial_overwrite_out2
        .type   resign_load_partial_overwrite_out2,@function
resign_load_partial_overwrite_out2:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function resign_load_partial_overwrite_out2, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacda   x0, x2
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      ldp     w10, w0, [x0]
        autdb   x0, x1
        ldp     w10, w0, [x0]
        pacda   x0, x2
        ret
        .size resign_load_partial_overwrite_out2, .-resign_load_partial_overwrite_out2

// Test that base register + offset register addressing mode is rejected.

        .globl  resign_load_reg_plus_reg
        .type   resign_load_reg_plus_reg,@function
resign_load_reg_plus_reg:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function resign_load_reg_plus_reg, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacda   x0, x2
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        autdb   x0, x1
        ldr     x3, [x0, x4]
        pacda   x0, x2
        ret
        .size resign_load_reg_plus_reg, .-resign_load_reg_plus_reg

        .globl  resign_load_reg_plus_reg_in2
        .type   resign_load_reg_plus_reg_in2,@function
resign_load_reg_plus_reg_in2:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function resign_load_reg_plus_reg_in2, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacda   x0, x2
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        autdb   x0, x1
        ldr     x3, [x4, x0]
        pacda   x0, x2
        ret
        .size resign_load_reg_plus_reg_in2, .-resign_load_reg_plus_reg_in2

        .globl  resign_load_unscaled_good
        .type   resign_load_unscaled_good,@function
resign_load_unscaled_good:
// CHECK-NOT: resign_load_unscaled_good
        autdb   x0, x1
        ldurb   w3, [x0, #-1]
        pacda   x0, x2
        ret
        .size resign_load_unscaled_good, .-resign_load_unscaled_good

// Any basic block can check at most one register using a multi-instruction
// pointer-checking sequence, but it can contain an arbitrary number of single-
// instruction pointer checks.

        .globl  many_checked_regs
        .type   many_checked_regs,@function
many_checked_regs:
// CHECK-NOT: many_checked_regs
        autdzb  x0
        autdzb  x1
        autdzb  x2
        b       1f
1:
        ldr     w3, [x0]  // single-instruction check
        ldr     w3, [x1]  // single-instruction check
        mov     x16, x2   // start of multi-instruction checker sequence
        xpacd   x16       // ...
        cmp     x2, x16   // ...
        b.eq    2f        // end of basic block
        brk     0xc473
2:
        pacdza  x0
        pacdza  x1
        pacdza  x2
        ret
        .size many_checked_regs, .-many_checked_regs

        .globl  main
        .type   main,@function
main:
        mov     x0, 0
        ret
        .size   main, .-main
