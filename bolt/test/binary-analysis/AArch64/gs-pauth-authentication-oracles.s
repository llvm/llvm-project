// RUN: %clang %cflags -march=armv8.3-a %s -o %t.exe
// RUN: llvm-bolt-binary-analysis --scanners=pacret %t.exe 2>&1 | FileCheck -check-prefix=PACRET %s
// RUN: llvm-bolt-binary-analysis --scanners=pauth  %t.exe 2>&1 | FileCheck %s

// The detection of compiler-generated explicit pointer checks is tested in
// gs-pauth-address-checks.s, for that reason only test here "dummy-load" and
// "high-bits-notbi" checkers, as the shortest examples of checkers that are
// detected per-instruction and per-BB.

// PACRET-NOT: authentication oracle found in function

        .text

        .type   sym,@function
sym:
        ret
        .size sym, .-sym

        .globl  callee
        .type   callee,@function
callee:
        ret
        .size callee, .-callee

        .globl  good_ret
        .type   good_ret,@function
good_ret:
// CHECK-NOT: good_ret
        autia   x0, x1
        ret     x0
        .size good_ret, .-good_ret

        .globl  good_call
        .type   good_call,@function
good_call:
// CHECK-NOT: good_call
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        autia   x0, x1
        blr     x0

        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size good_call, .-good_call

        .globl  good_branch
        .type   good_branch,@function
good_branch:
// CHECK-NOT: good_branch
        autia   x0, x1
        br      x0
        .size good_branch, .-good_branch

        .globl  good_load_other_reg
        .type   good_load_other_reg,@function
good_load_other_reg:
// CHECK-NOT: good_load_other_reg
        autia   x0, x1
        ldr     x2, [x0]
        ret
        .size good_load_other_reg, .-good_load_other_reg

        .globl  good_load_same_reg
        .type   good_load_same_reg,@function
good_load_same_reg:
// CHECK-NOT: good_load_same_reg
        autia   x0, x1
        ldr     x0, [x0]
        ret
        .size good_load_same_reg, .-good_load_same_reg

        .globl  good_explicit_check
        .type   good_explicit_check,@function
good_explicit_check:
// CHECK-NOT: good_explicit_check
        autia   x0, x1
        eor     x16, x0, x0, lsl #1
        tbz     x16, #62, 1f
        brk     0xc470
1:
        ret
        .size good_explicit_check, .-good_explicit_check

        .globl  bad_unchecked
        .type   bad_unchecked,@function
bad_unchecked:
// CHECK-LABEL: GS-PAUTH: authentication oracle found in function bad_unchecked, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      autia   x0, x1
// CHECK-NEXT:  The 0 instructions that leak the affected registers are:
        autia   x0, x1
        ret
        .size bad_unchecked, .-bad_unchecked

        .globl  bad_leaked_to_subroutine
        .type   bad_leaked_to_subroutine,@function
bad_leaked_to_subroutine:
// CHECK-LABEL: GS-PAUTH: authentication oracle found in function bad_leaked_to_subroutine, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      autia   x0, x1
// CHECK-NEXT:  The 1 instructions that leak the affected registers are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      bl      callee
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   paciasp
// CHECK-NEXT:  {{[0-9a-f]+}}:   stp     x29, x30, [sp, #-0x10]!
// CHECK-NEXT:  {{[0-9a-f]+}}:   mov     x29, sp
// CHECK-NEXT:  {{[0-9a-f]+}}:   autia   x0, x1
// CHECK-NEXT:  {{[0-9a-f]+}}:   bl      callee
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldr     x2, [x0]
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  {{[0-9a-f]+}}:   autiasp
// CHECK-NEXT:  {{[0-9a-f]+}}:   ret
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        autia   x0, x1
        bl      callee
        ldr     x2, [x0]

        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size bad_leaked_to_subroutine, .-bad_leaked_to_subroutine

        .globl  bad_unknown_usage_read
        .type   bad_unknown_usage_read,@function
bad_unknown_usage_read:
// CHECK-LABEL: GS-PAUTH: authentication oracle found in function bad_unknown_usage_read, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      autia   x0, x1
// CHECK-NEXT:  The 1 instructions that leak the affected registers are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      mul     x3, x0, x1
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   autia   x0, x1
// CHECK-NEXT:  {{[0-9a-f]+}}:   mul     x3, x0, x1
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldr     x2, [x0]
// CHECK-NEXT:  {{[0-9a-f]+}}:   ret
        autia   x0, x1
        // Registers are not accessible to an attacker under Pointer
        // Authentication threat model, until spilled to memory.
        // Thus, reporting the below MUL instruction is a false positive, since
        // the next LDR instruction prevents any possible spilling of x3 unless
        // the authentication succeeded. Though, rejecting anything except for
        // a closed list of instruction types is the intended behavior of the
        // analysis, so this false positive is by design.
        mul     x3, x0, x1
        ldr     x2, [x0]
        ret
        .size bad_unknown_usage_read, .-bad_unknown_usage_read

        .globl  bad_store_to_memory_and_wait
        .type   bad_store_to_memory_and_wait,@function
bad_store_to_memory_and_wait:
// CHECK-LABEL: GS-PAUTH: authentication oracle found in function bad_store_to_memory_and_wait, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      autia   x0, x1
// CHECK-NEXT:  The 1 instructions that leak the affected registers are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      str     x0, [x3]
        autia   x0, x1
        cbz     x3, 2f
        str     x0, [x3]
1:
        // The thread performs a time-consuming computation while the result of
        // authentication is accessible in memory.
        nop
2:
        ldr     x2, [x0]
        ret
        .size bad_store_to_memory_and_wait, .-bad_store_to_memory_and_wait

// FIXME: Known false negative: if no return instruction is reachable from a
//        program point (this probably implies an infinite loop), such
//        instruction cannot be detected as an authentication oracle.
        .globl  bad_store_to_memory_and_hang
        .type   bad_store_to_memory_and_hang,@function
bad_store_to_memory_and_hang:
// CHECK-NOT: bad_store_to_memory_and_hang
        autia   x0, x1
        cbz     x3, 2f
        str     x0, [x3]
1:
        // The thread loops indefinitely while the result of authentication
        // is accessible in memory.
        b       1b
2:
        ldr     x2, [x0]
        ret
        .size bad_store_to_memory_and_hang, .-bad_store_to_memory_and_hang

        .globl  bad_unknown_usage_subreg_read
        .type   bad_unknown_usage_subreg_read,@function
bad_unknown_usage_subreg_read:
// CHECK-LABEL: GS-PAUTH: authentication oracle found in function bad_unknown_usage_subreg_read, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      autia   x0, x1
// CHECK-NEXT:  The 1 instructions that leak the affected registers are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      mul     w3, w0, w1
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   autia   x0, x1
// CHECK-NEXT:  {{[0-9a-f]+}}:   mul     w3, w0, w1
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldr     x2, [x0]
// CHECK-NEXT:  {{[0-9a-f]+}}:   ret
        autia   x0, x1
        mul     w3, w0, w1
        ldr     x2, [x0]
        ret
        .size bad_unknown_usage_subreg_read, .-bad_unknown_usage_subreg_read

        .globl  bad_unknown_usage_update
        .type   bad_unknown_usage_update,@function
bad_unknown_usage_update:
// CHECK-LABEL: GS-PAUTH: authentication oracle found in function bad_unknown_usage_update, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      autia   x0, x1
// CHECK-NEXT:  The 1 instructions that leak the affected registers are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      movk    x0, #0x2a, lsl #16
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   autia   x0, x1
// CHECK-NEXT:  {{[0-9a-f]+}}:   movk    x0, #0x2a, lsl #16
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldr     x2, [x0]
// CHECK-NEXT:  {{[0-9a-f]+}}:   ret
        autia   x0, x1
        movk    x0, #42, lsl #16 // does not overwrite x0 completely
        ldr     x2, [x0]
        ret
        .size bad_unknown_usage_update, .-bad_unknown_usage_update

        .globl  good_overwrite_with_constant
        .type   good_overwrite_with_constant,@function
good_overwrite_with_constant:
// CHECK-NOT: good_overwrite_with_constant
        autia   x0, x1
        mov     x0, #42
        ret
        .size good_overwrite_with_constant, .-good_overwrite_with_constant

// Overwriting sensitive data by instructions with unmodelled side-effects is
// explicitly rejected, even though this particular MRS is safe.
        .globl  bad_overwrite_with_side_effects
        .type   bad_overwrite_with_side_effects,@function
bad_overwrite_with_side_effects:
// CHECK-LABEL: GS-PAUTH: authentication oracle found in function bad_overwrite_with_side_effects, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      autia   x0, x1
// CHECK-NEXT:  The 0 instructions that leak the affected registers are:
        autia   x0, x1
        mrs     x0, CTR_EL0
        ret
        .size bad_overwrite_with_side_effects, .-bad_overwrite_with_side_effects

// Here the new value written by MUL to x0 is completely unrelated to the result
// of authentication, so this is a false positive.
// FIXME: Can/should we generalize overwriting by constant to handle such cases?
        .globl  good_unknown_overwrite
        .type   good_unknown_overwrite,@function
good_unknown_overwrite:
// CHECK-LABEL: GS-PAUTH: authentication oracle found in function good_unknown_overwrite, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      autia   x0, x1
// CHECK-NEXT:  The 0 instructions that leak the affected registers are:
        autia   x0, x1
        mul     x0, x1, x2
        ret
        .size good_unknown_overwrite, .-good_unknown_overwrite

// This is a false positive: when a general-purpose register is written to as
// a 32-bit register, its top 32 bits are zeroed, but according to LLVM
// representation, the instruction only overwrites the Wn register.
        .globl  good_wreg_overwrite
        .type   good_wreg_overwrite,@function
good_wreg_overwrite:
// CHECK-LABEL: GS-PAUTH: authentication oracle found in function good_wreg_overwrite, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      autia   x0, x1
        autia   x0, x1
        mov     w0, #42
        ret
        .size good_wreg_overwrite, .-good_wreg_overwrite

        .globl  good_address_arith
        .type   good_address_arith,@function
good_address_arith:
// CHECK-NOT: good_address_arith
        autia   x0, x1

        add     x1, x0, #8
        sub     x2, x1, #16
        mov     x3, x2

        ldr     x4, [x3]
        mov     x0, #0
        mov     x1, #0
        mov     x2, #0

        ret
        .size good_address_arith, .-good_address_arith

        .globl  good_ret_multi_bb
        .type   good_ret_multi_bb,@function
good_ret_multi_bb:
// CHECK-NOT: good_ret_multi_bb
        autia   x0, x1
        cbz     x1, 1f
        nop
1:
        ret     x0
        .size good_ret_multi_bb, .-good_ret_multi_bb

        .globl  good_call_multi_bb
        .type   good_call_multi_bb,@function
good_call_multi_bb:
// CHECK-NOT: good_call_multi_bb
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        autia   x0, x1
        cbz     x1, 1f
        nop
1:
        blr     x0
        cbz     x1, 2f
        nop
2:
        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size good_call_multi_bb, .-good_call_multi_bb

        .globl  good_branch_multi_bb
        .type   good_branch_multi_bb,@function
good_branch_multi_bb:
// CHECK-NOT: good_branch_multi_bb
        autia   x0, x1
        cbz     x1, 1f
        nop
1:
        br      x0
        .size good_branch_multi_bb, .-good_branch_multi_bb

        .globl  good_load_other_reg_multi_bb
        .type   good_load_other_reg_multi_bb,@function
good_load_other_reg_multi_bb:
// CHECK-NOT: good_load_other_reg_multi_bb
        autia   x0, x1
        cbz     x1, 1f
        nop
1:
        ldr     x2, [x0]
        cbz     x1, 2f
        nop
2:
        ret
        .size good_load_other_reg_multi_bb, .-good_load_other_reg_multi_bb

        .globl  good_load_same_reg_multi_bb
        .type   good_load_same_reg_multi_bb,@function
good_load_same_reg_multi_bb:
// CHECK-NOT: good_load_same_reg_multi_bb
        autia   x0, x1
        cbz     x1, 1f
        nop
1:
        ldr     x0, [x0]
        cbz     x1, 2f
        nop
2:
        ret
        .size good_load_same_reg_multi_bb, .-good_load_same_reg_multi_bb

        .globl  good_explicit_check_multi_bb
        .type   good_explicit_check_multi_bb,@function
good_explicit_check_multi_bb:
// CHECK-NOT: good_explicit_check_multi_bb
        autia   x0, x1
        cbz     x1, 1f
        nop
1:
        eor     x16, x0, x0, lsl #1
        tbz     x16, #62, 2f
        brk     0xc470
2:
        cbz     x1, 3f
        nop
3:
        ret
        .size good_explicit_check_multi_bb, .-good_explicit_check_multi_bb

        .globl  bad_unchecked_multi_bb
        .type   bad_unchecked_multi_bb,@function
bad_unchecked_multi_bb:
// CHECK-LABEL: GS-PAUTH: authentication oracle found in function bad_unchecked_multi_bb, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      autia   x0, x1
// CHECK-NEXT:  The 0 instructions that leak the affected registers are:
        autia   x0, x1
        cbz     x1, 1f
        ldr     x2, [x0]
1:
        ret
        .size bad_unchecked_multi_bb, .-bad_unchecked_multi_bb

        .globl  bad_leaked_to_subroutine_multi_bb
        .type   bad_leaked_to_subroutine_multi_bb,@function
bad_leaked_to_subroutine_multi_bb:
// CHECK-LABEL: GS-PAUTH: authentication oracle found in function bad_leaked_to_subroutine_multi_bb, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      autia   x0, x1
// CHECK-NEXT:  The 1 instructions that leak the affected registers are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      bl      callee
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        autia   x0, x1
        cbz     x1, 1f
        ldr     x2, [x0]
1:
        bl      callee
        ldr     x2, [x0]

        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size bad_leaked_to_subroutine_multi_bb, .-bad_leaked_to_subroutine_multi_bb

        .globl  bad_unknown_usage_read_multi_bb
        .type   bad_unknown_usage_read_multi_bb,@function
bad_unknown_usage_read_multi_bb:
// CHECK-LABEL: GS-PAUTH: authentication oracle found in function bad_unknown_usage_read_multi_bb, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      autia   x0, x1
// CHECK-NEXT:  The 1 instructions that leak the affected registers are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      mul     x3, x0, x1
        autia   x0, x1
        cbz     x3, 1f
        mul     x3, x0, x1
1:
        ldr     x2, [x0]
        ret
        .size bad_unknown_usage_read_multi_bb, .-bad_unknown_usage_read_multi_bb

        .globl  bad_unknown_usage_subreg_read_multi_bb
        .type   bad_unknown_usage_subreg_read_multi_bb,@function
bad_unknown_usage_subreg_read_multi_bb:
// CHECK-LABEL: GS-PAUTH: authentication oracle found in function bad_unknown_usage_subreg_read_multi_bb, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      autia   x0, x1
// CHECK-NEXT:  The 1 instructions that leak the affected registers are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      mul     w3, w0, w1
        autia   x0, x1
        cbz     x3, 1f
        mul     w3, w0, w1
1:
        ldr     x2, [x0]
        ret
        .size bad_unknown_usage_subreg_read_multi_bb, .-bad_unknown_usage_subreg_read_multi_bb

        .globl  bad_unknown_usage_update_multi_bb
        .type   bad_unknown_usage_update_multi_bb,@function
bad_unknown_usage_update_multi_bb:
// CHECK-LABEL: GS-PAUTH: authentication oracle found in function bad_unknown_usage_update_multi_bb, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      autia   x0, x1
// CHECK-NEXT:  The 1 instructions that leak the affected registers are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      movk    x0, #0x2a, lsl #16
        autia   x0, x1
        cbz     x3, 1f
        movk    x0, #42, lsl #16  // does not overwrite x0 completely
1:
        ldr     x2, [x0]
        ret
        .size bad_unknown_usage_update_multi_bb, .-bad_unknown_usage_update_multi_bb

        .globl  good_overwrite_with_constant_multi_bb
        .type   good_overwrite_with_constant_multi_bb,@function
good_overwrite_with_constant_multi_bb:
// CHECK-NOT: good_overwrite_with_constant_multi_bb
        autia   x0, x1
        cbz     x3, 1f
1:
        mov     x0, #42
        ret
        .size good_overwrite_with_constant_multi_bb, .-good_overwrite_with_constant_multi_bb

        .globl  good_address_arith_multi_bb
        .type   good_address_arith_multi_bb,@function
good_address_arith_multi_bb:
// CHECK-NOT: good_address_arith_multi_bb
        autia   x0, x1
        cbz     x3, 1f

        add     x1, x0, #8
        sub     x2, x1, #16
        mov     x0, x2

        mov     x1, #0
        mov     x2, #0
1:
        ldr     x3, [x0]
        ret
        .size good_address_arith_multi_bb, .-good_address_arith_multi_bb

        .globl  good_ret_nocfg
        .type   good_ret_nocfg,@function
good_ret_nocfg:
// CHECK-NOT: good_ret_nocfg
        adr     x2, 1f
        br      x2
1:
        autia   x0, x1

        ret     x0
        .size good_ret_nocfg, .-good_ret_nocfg

        .globl  good_call_nocfg
        .type   good_call_nocfg,@function
good_call_nocfg:
// CHECK-NOT: good_call_nocfg
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        adr     x2, 1f
        br      x2
1:
        autia   x0, x1
        blr     x0

        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size good_call_nocfg, .-good_call_nocfg

        .globl  good_branch_nocfg
        .type   good_branch_nocfg,@function
good_branch_nocfg:
// CHECK-NOT: good_branch_nocfg
        adr     x2, 1f
        br      x2
1:
        autia   x0, x1
        br      x0
        .size good_branch_nocfg, .-good_branch_nocfg

        .globl  good_load_other_reg_nocfg
        .type   good_load_other_reg_nocfg,@function
good_load_other_reg_nocfg:
// CHECK-NOT: good_load_other_reg_nocfg
        adr     x2, 1f
        br      x2
1:
        autia   x0, x1
        ldr     x2, [x0]

        ret
        .size good_load_other_reg_nocfg, .-good_load_other_reg_nocfg

        .globl  good_load_same_reg_nocfg
        .type   good_load_same_reg_nocfg,@function
good_load_same_reg_nocfg:
// CHECK-NOT: good_load_same_reg_nocfg
        adr     x2, 1f
        br      x2
1:
        autia   x0, x1
        ldr     x0, [x0]

        ret
        .size good_load_same_reg_nocfg, .-good_load_same_reg_nocfg

// FIXME: Multi-instruction checker sequences are not supported without CFG.

        .globl  bad_unchecked_nocfg
        .type   bad_unchecked_nocfg,@function
bad_unchecked_nocfg:
// CHECK-LABEL: GS-PAUTH: authentication oracle found in function bad_unchecked_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      autia   x0, x1
// CHECK-NEXT:  The 0 instructions that leak the affected registers are:
        adr     x2, 1f
        br      x2
1:
        autia   x0, x1

        ret
        .size bad_unchecked_nocfg, .-bad_unchecked_nocfg

        .globl  bad_leaked_to_subroutine_nocfg
        .type   bad_leaked_to_subroutine_nocfg,@function
bad_leaked_to_subroutine_nocfg:
// CHECK-LABEL: GS-PAUTH: authentication oracle found in function bad_leaked_to_subroutine_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      autia   x0, x1
// CHECK-NEXT:  The 1 instructions that leak the affected registers are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      bl      callee # Offset: 24
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        adr     x2, 1f
        br      x2
1:
        autia   x0, x1
        bl      callee
        ldr     x2, [x0]

        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size bad_leaked_to_subroutine_nocfg, .-bad_leaked_to_subroutine_nocfg

        .globl  bad_unknown_usage_read_nocfg
        .type   bad_unknown_usage_read_nocfg,@function
bad_unknown_usage_read_nocfg:
// CHECK-LABEL: GS-PAUTH: authentication oracle found in function bad_unknown_usage_read_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      autia   x0, x1
// CHECK-NEXT:  The 1 instructions that leak the affected registers are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      mul     x3, x0, x1
        adr     x2, 1f
        br      x2
1:
        autia   x0, x1
        mul     x3, x0, x1
        ldr     x2, [x0]

        ret
        .size bad_unknown_usage_read_nocfg, .-bad_unknown_usage_read_nocfg

        .globl  bad_unknown_usage_subreg_read_nocfg
        .type   bad_unknown_usage_subreg_read_nocfg,@function
bad_unknown_usage_subreg_read_nocfg:
// CHECK-LABEL: GS-PAUTH: authentication oracle found in function bad_unknown_usage_subreg_read_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      autia   x0, x1
// CHECK-NEXT:  The 1 instructions that leak the affected registers are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      mul     w3, w0, w1
        adr     x2, 1f
        br      x2
1:
        autia   x0, x1
        mul     w3, w0, w1
        ldr     x2, [x0]

        ret
        .size bad_unknown_usage_subreg_read_nocfg, .-bad_unknown_usage_subreg_read_nocfg

        .globl  bad_unknown_usage_update_nocfg
        .type   bad_unknown_usage_update_nocfg,@function
bad_unknown_usage_update_nocfg:
// CHECK-LABEL: GS-PAUTH: authentication oracle found in function bad_unknown_usage_update_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      autia   x0, x1
// CHECK-NEXT:  The 1 instructions that leak the affected registers are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      movk    x0, #0x2a, lsl #16
        adr     x2, 1f
        br      x2
1:
        autia   x0, x1
        movk    x0, #42, lsl #16  // does not overwrite x0 completely
        ldr     x2, [x0]

        ret
        .size bad_unknown_usage_update_nocfg, .-bad_unknown_usage_update_nocfg

        .globl  good_overwrite_with_constant_nocfg
        .type   good_overwrite_with_constant_nocfg,@function
good_overwrite_with_constant_nocfg:
// CHECK-NOT: good_overwrite_with_constant_nocfg
        adr     x2, 1f
        br      x2
1:
        autia   x0, x1
        mov     x0, #42

        ret
        .size good_overwrite_with_constant_nocfg, .-good_overwrite_with_constant_nocfg

        .globl  good_address_arith_nocfg
        .type   good_address_arith_nocfg,@function
good_address_arith_nocfg:
// CHECK-NOT: good_address_arith_nocfg
        adr     x2, 1f
        br      x2
1:
        autia   x0, x1
        add     x1, x0, #8
        sub     x2, x1, #16
        mov     x3, x2

        ldr     x4, [x3]
        mov     x0, #0
        mov     x1, #0
        mov     x2, #0

        ret
        .size good_address_arith_nocfg, .-good_address_arith_nocfg

        .globl  good_explicit_check_unrelated_reg
        .type   good_explicit_check_unrelated_reg,@function
good_explicit_check_unrelated_reg:
// CHECK-NOT: good_explicit_check_unrelated_reg
        autia   x2, x3    // One of possible execution paths after this instruction
                          // ends at BRK below, thus BRK used as a trap instruction
                          // should formally "check everything" not to introduce
                          // false-positive here.
        autia   x0, x1
        eor     x16, x0, x0, lsl #1
        tbz     x16, #62, 1f
        brk     0xc470
1:
        ldr     x4, [x2]  // Right before this instruction X2 is checked - this
                          // should be propagated to the basic block ending with
                          // TBZ instruction above.
        ret
        .size good_explicit_check_unrelated_reg, .-good_explicit_check_unrelated_reg

// The last BB (in layout order) is processed first by the data-flow analysis.
// Its initial state is usually filled in a special way (because it ends with
// `ret` instruction), and then affects the state propagated to the other BBs
// Thus, the case of the last instruction in a function being a jump somewhere
// in the middle is special.

        .globl  good_no_ret_from_last_bb
        .type   good_no_ret_from_last_bb,@function
good_no_ret_from_last_bb:
// CHECK-NOT: good_no_ret_from_last_bb
        paciasp
        autiasp     // authenticates LR
        b       2f
1:
        ret
2:
        b       1b  // LR is dereferenced by `ret`, which is executed next
        .size good_no_ret_from_last_bb, .-good_no_ret_from_last_bb

        .globl  bad_no_ret_from_last_bb
        .type   bad_no_ret_from_last_bb,@function
bad_no_ret_from_last_bb:
// CHECK-LABEL: GS-PAUTH: authentication oracle found in function bad_no_ret_from_last_bb, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      autiasp
// CHECK-NEXT:  The 0 instructions that leak the affected registers are:
        paciasp
        autiasp     // authenticates LR
        b       2f
1:
        ret     x0
2:
        b       1b  // X0 (but not LR) is dereferenced by `ret x0`
        .size bad_no_ret_from_last_bb, .-bad_no_ret_from_last_bb

// Test that combined auth+something instructions are not reported as
// authentication oracles.

        .globl  inst_retaa
        .type   inst_retaa,@function
inst_retaa:
// CHECK-NOT: inst_retaa
        paciasp
        retaa
        .size inst_retaa, .-inst_retaa

        .globl  inst_blraa
        .type   inst_blraa,@function
inst_blraa:
// CHECK-NOT: inst_blraa
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        blraa   x0, x1

        ldp     x29, x30, [sp], #16
        retaa
        .size inst_blraa, .-inst_blraa

        .globl  inst_braa
        .type   inst_braa,@function
inst_braa:
// CHECK-NOT: inst_braa
        braa    x0, x1
        .size inst_braa, .-inst_braa

        .globl  inst_ldraa_no_wb
        .type   inst_ldraa_no_wb,@function
inst_ldraa_no_wb:
// CHECK-NOT: inst_ldraa_no_wb
        ldraa   x1, [x0]
        ret
        .size inst_ldraa_no_wb, .-inst_ldraa_no_wb

        .globl  inst_ldraa_wb
        .type   inst_ldraa_wb,@function
inst_ldraa_wb:
// CHECK-NOT: inst_ldraa_wb
        ldraa   x1, [x0]!
        ret
        .size inst_ldraa_wb, .-inst_ldraa_wb

        .globl  main
        .type   main,@function
main:
        mov     x0, 0
        ret
        .size   main, .-main
