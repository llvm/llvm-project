// RUN: %clang %cflags -march=armv8.3-a+pauth-lr -Wl,--no-relax %s -o %t.exe
// RUN: llvm-bolt-binary-analysis --scanners=pacret %t.exe 2>&1 | FileCheck -check-prefix=PACRET %s
// RUN: llvm-bolt-binary-analysis --scanners=pauth  %t.exe 2>&1 | FileCheck %s

// The detection of compiler-generated explicit pointer checks is tested in
// gs-pauth-address-checks.s, for that reason only test here "dummy-load" and
// "high-bits-notbi" checkers, as the shortest examples of checkers that are
// detected per-instruction and per-BB.

// PACRET-NOT: signing oracle found in function

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

// Test transitions between register states: none, safe-to-dereference (s-t-d), trusted:
// * trusted right away: safe address materialization
// * trusted as checked s-t-d: two variants of checks
// * untrusted: s-t-d, but not checked
// * untrusted: not s-t-d, but checked
// * untrusted: not even s-t-d - from arg and from memory
// * untrusted: {subreg clobbered, function called} X {between address materialization and use, between auth and check, between check and use}
// * untrusted: first checked then auted, auted then auted, checked then checked

        .globl  good_sign_addr_mat
        .type   good_sign_addr_mat,@function
good_sign_addr_mat:
// CHECK-NOT: good_sign_addr_mat
        adr     x0, sym
        pacda   x0, x1
        ret
        .size good_sign_addr_mat, .-good_sign_addr_mat

        .globl  good_sign_auted_checked_ldr
        .type   good_sign_auted_checked_ldr,@function
good_sign_auted_checked_ldr:
// CHECK-NOT: good_sign_auted_checked_ldr
        autda   x0, x2
        ldr     x2, [x0]
        pacda   x0, x1
        ret
        .size good_sign_auted_checked_ldr, .-good_sign_auted_checked_ldr

        .globl  good_sign_auted_checked_brk
        .type   good_sign_auted_checked_brk,@function
good_sign_auted_checked_brk:
// CHECK-NOT: good_sign_auted_checked_brk
        autda   x0, x2
        eor     x16, x0, x0, lsl #1
        tbz     x16, #62, 1f
        brk     0xc472
1:
        pacda   x0, x1
        ret
        .size good_sign_auted_checked_brk, .-good_sign_auted_checked_brk

        .globl  bad_sign_authed_unchecked
        .type   bad_sign_authed_unchecked,@function
bad_sign_authed_unchecked:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function bad_sign_authed_unchecked, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:     pacda   x0, x1
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        autda   x0, x2
        pacda   x0, x1
        ret
        .size bad_sign_authed_unchecked, .-bad_sign_authed_unchecked

        .globl  bad_sign_checked_not_auted
        .type   bad_sign_checked_not_auted,@function
bad_sign_checked_not_auted:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function bad_sign_checked_not_auted, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:     pacda   x0, x1
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        ldr     x2, [x0]
        pacda   x0, x1
        ret
        .size bad_sign_checked_not_auted, .-bad_sign_checked_not_auted

        .globl  bad_sign_plain_arg
        .type   bad_sign_plain_arg,@function
bad_sign_plain_arg:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function bad_sign_plain_arg, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:     pacda   x0, x1
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        pacda   x0, x1
        ret
        .size bad_sign_plain_arg, .-bad_sign_plain_arg

        .globl  bad_sign_plain_mem
        .type   bad_sign_plain_mem,@function
bad_sign_plain_mem:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function bad_sign_plain_mem, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:     pacda   x0, x1
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      ldr     x0, [x1]
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldr     x0, [x1]
// CHECK-NEXT:  {{[0-9a-f]+}}:   pacda   x0, x1
// CHECK-NEXT:  {{[0-9a-f]+}}:   ret
        ldr     x0, [x1]
        pacda   x0, x1
        ret
        .size bad_sign_plain_mem, .-bad_sign_plain_mem

        .globl  bad_clobber_between_addr_mat_and_use
        .type   bad_clobber_between_addr_mat_and_use,@function
bad_clobber_between_addr_mat_and_use:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function bad_clobber_between_addr_mat_and_use, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:     pacda   x0, x1
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      mov     w0, w3
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   adr     x0, "sym/1"
// CHECK-NEXT:  {{[0-9a-f]+}}:   mov     w0, w3
// CHECK-NEXT:  {{[0-9a-f]+}}:   pacda   x0, x1
// CHECK-NEXT:  {{[0-9a-f]+}}:   ret
        adr     x0, sym
        mov     w0, w3
        pacda   x0, x1
        ret
        .size bad_clobber_between_addr_mat_and_use, .-bad_clobber_between_addr_mat_and_use

        .globl  bad_clobber_between_auted_and_checked
        .type   bad_clobber_between_auted_and_checked,@function
bad_clobber_between_auted_and_checked:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function bad_clobber_between_auted_and_checked, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:     pacda   x0, x1
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      mov     w0, w3
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   autda   x0, x2
// CHECK-NEXT:  {{[0-9a-f]+}}:   mov     w0, w3
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldr     x2, [x0]
// CHECK-NEXT:  {{[0-9a-f]+}}:   pacda   x0, x1
// CHECK-NEXT:  {{[0-9a-f]+}}:   ret
        autda   x0, x2
        mov     w0, w3
        ldr     x2, [x0]
        pacda   x0, x1
        ret
        .size bad_clobber_between_auted_and_checked, .-bad_clobber_between_auted_and_checked

        .globl  bad_clobber_between_checked_and_used
        .type   bad_clobber_between_checked_and_used,@function
bad_clobber_between_checked_and_used:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function bad_clobber_between_checked_and_used, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:     pacda   x0, x1
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      mov     w0, w3
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   autda   x0, x2
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldr     x2, [x0]
// CHECK-NEXT:  {{[0-9a-f]+}}:   mov     w0, w3
// CHECK-NEXT:  {{[0-9a-f]+}}:   pacda   x0, x1
// CHECK-NEXT:  {{[0-9a-f]+}}:   ret
        autda   x0, x2
        ldr     x2, [x0]
        mov     w0, w3
        pacda   x0, x1
        ret
        .size bad_clobber_between_checked_and_used, .-bad_clobber_between_checked_and_used

        .globl  bad_call_between_addr_mat_and_use
        .type   bad_call_between_addr_mat_and_use,@function
bad_call_between_addr_mat_and_use:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function bad_call_between_addr_mat_and_use, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:     pacda   x0, x1
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      bl      callee
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   paciasp
// CHECK-NEXT:  {{[0-9a-f]+}}:   stp     x29, x30, [sp, #-0x10]!
// CHECK-NEXT:  {{[0-9a-f]+}}:   mov     x29, sp
// CHECK-NEXT:  {{[0-9a-f]+}}:   adr     x0, "sym/1"
// CHECK-NEXT:  {{[0-9a-f]+}}:   bl      callee
// CHECK-NEXT:  {{[0-9a-f]+}}:   pacda   x0, x1
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  {{[0-9a-f]+}}:   autiasp
// CHECK-NEXT:  {{[0-9a-f]+}}:   ret
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        adr     x0, sym
        bl      callee
        pacda   x0, x1

        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size bad_call_between_addr_mat_and_use, .-bad_call_between_addr_mat_and_use

        .globl  bad_call_between_auted_and_checked
        .type   bad_call_between_auted_and_checked,@function
bad_call_between_auted_and_checked:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function bad_call_between_auted_and_checked, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:     pacda   x0, x1
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      bl      callee
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   paciasp
// CHECK-NEXT:  {{[0-9a-f]+}}:   stp     x29, x30, [sp, #-0x10]!
// CHECK-NEXT:  {{[0-9a-f]+}}:   mov     x29, sp
// CHECK-NEXT:  {{[0-9a-f]+}}:   autda   x0, x2
// CHECK-NEXT:  {{[0-9a-f]+}}:   bl      callee
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldr     x2, [x0]
// CHECK-NEXT:  {{[0-9a-f]+}}:   pacda   x0, x1
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  {{[0-9a-f]+}}:   autiasp
// CHECK-NEXT:  {{[0-9a-f]+}}:   ret
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        autda   x0, x2
        bl      callee
        ldr     x2, [x0]
        pacda   x0, x1

        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size bad_call_between_auted_and_checked, .-bad_call_between_auted_and_checked

        .globl  bad_call_between_checked_and_used
        .type   bad_call_between_checked_and_used,@function
bad_call_between_checked_and_used:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function bad_call_between_checked_and_used, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:     pacda   x0, x1
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      bl      callee
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   paciasp
// CHECK-NEXT:  {{[0-9a-f]+}}:   stp     x29, x30, [sp, #-0x10]!
// CHECK-NEXT:  {{[0-9a-f]+}}:   mov     x29, sp
// CHECK-NEXT:  {{[0-9a-f]+}}:   autda   x0, x2
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldr     x2, [x0]
// CHECK-NEXT:  {{[0-9a-f]+}}:   bl      callee
// CHECK-NEXT:  {{[0-9a-f]+}}:   pacda   x0, x1
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  {{[0-9a-f]+}}:   autiasp
// CHECK-NEXT:  {{[0-9a-f]+}}:   ret
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        autda   x0, x2
        ldr     x2, [x0]
        bl      callee
        pacda   x0, x1

        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size bad_call_between_checked_and_used, .-bad_call_between_checked_and_used

        .globl  bad_transition_check_then_auth
        .type   bad_transition_check_then_auth,@function
bad_transition_check_then_auth:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function bad_transition_check_then_auth, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:     pacda   x0, x1
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        ldr     x2, [x0]
        autda   x0, x2
        pacda   x0, x1
        ret
        .size bad_transition_check_then_auth, .-bad_transition_check_then_auth

        .globl  bad_transition_auth_then_auth
        .type   bad_transition_auth_then_auth,@function
bad_transition_auth_then_auth:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function bad_transition_auth_then_auth, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:     pacda   x0, x1
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        autda   x0, x2
        autda   x0, x2
        pacda   x0, x1
        ret
        .size bad_transition_auth_then_auth, .-bad_transition_auth_then_auth

        .globl  bad_transition_check_then_check
        .type   bad_transition_check_then_check,@function
bad_transition_check_then_check:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function bad_transition_check_then_check, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:     pacda   x0, x1
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        ldr     x2, [x0]
        ldr     x2, [x0]
        pacda   x0, x1
        ret
        .size bad_transition_check_then_check, .-bad_transition_check_then_check

// Multi-BB test cases.

// Test state propagation across multiple basic blocks.
// Test transitions between register states: none, safe-to-dereference (s-t-d), trusted:
// * trusted right away: safe address materialization
// * trusted as checked s-t-d: two variants of checks
// * untrusted: s-t-d, but not *always* checked
// * untrusted: not *always* s-t-d, but checked
// * untrusted: not even s-t-d - from arg and from memory
// * untrusted: subreg clobbered - between address materialization and use, between auth and check, between check and use
// * trusted in both predecessors but for different reasons
//   (the one due to address materialization and the other due to s-t-d then checked)
// * untrusted: auted in one predecessor, checked in the other

        .globl  good_sign_addr_mat_multi_bb
        .type   good_sign_addr_mat_multi_bb,@function
good_sign_addr_mat_multi_bb:
// CHECK-NOT: good_sign_addr_mat_multi_bb
        adr     x0, sym
        cbz     x3, 1f
        nop
1:
        pacda   x0, x1
        ret
        .size good_sign_addr_mat_multi_bb, .-good_sign_addr_mat_multi_bb

        .globl  good_sign_auted_checked_ldr_multi_bb
        .type   good_sign_auted_checked_ldr_multi_bb,@function
good_sign_auted_checked_ldr_multi_bb:
// CHECK-NOT: good_sign_auted_checked_ldr_multi_bb
        autda   x0, x2
        cbz     x3, 1f
        nop
1:
        ldr     x2, [x0]
        cbz     x4, 2f
        nop
2:
        pacda   x0, x1
        ret
        .size good_sign_auted_checked_ldr_multi_bb, .-good_sign_auted_checked_ldr_multi_bb

        .globl  good_sign_auted_checked_brk_multi_bb
        .type   good_sign_auted_checked_brk_multi_bb,@function
good_sign_auted_checked_brk_multi_bb:
// CHECK-NOT: good_sign_auted_checked_brk_multi_bb
        autda   x0, x2
        cbz     x3, 1f
        nop
1:
        eor     x16, x0, x0, lsl #1
        tbz     x16, #62, 2f
        brk     0xc472
2:
        cbz     x4, 3f
        nop
3:
        pacda   x0, x1
        ret
        .size good_sign_auted_checked_brk_multi_bb, .-good_sign_auted_checked_brk_multi_bb

        .globl  bad_sign_authed_unchecked_multi_bb
        .type   bad_sign_authed_unchecked_multi_bb,@function
bad_sign_authed_unchecked_multi_bb:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function bad_sign_authed_unchecked_multi_bb, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:     pacda   x0, x1
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        autda   x0, x2
        cbz     x3, 1f
        ldr     x2, [x0]
1:
        pacda   x0, x1
        ret
        .size bad_sign_authed_unchecked_multi_bb, .-bad_sign_authed_unchecked_multi_bb

        .globl  bad_sign_checked_not_auted_multi_bb
        .type   bad_sign_checked_not_auted_multi_bb,@function
bad_sign_checked_not_auted_multi_bb:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function bad_sign_checked_not_auted_multi_bb, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:     pacda   x0, x1
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        cbz     x3, 1f
        autda   x0, x2
1:
        ldr     x2, [x0]
        pacda   x0, x1
        ret
        .size bad_sign_checked_not_auted_multi_bb, .-bad_sign_checked_not_auted_multi_bb

        .globl  bad_sign_plain_arg_multi_bb
        .type   bad_sign_plain_arg_multi_bb,@function
bad_sign_plain_arg_multi_bb:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function bad_sign_plain_arg_multi_bb, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:     pacda   x0, x1
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        cbz     x3, 1f
        autda   x0, x2
        ldr     x2, [x0]
1:
        pacda   x0, x1
        ret
        .size bad_sign_plain_arg_multi_bb, .-bad_sign_plain_arg_multi_bb

        .globl  bad_sign_plain_mem_multi_bb
        .type   bad_sign_plain_mem_multi_bb,@function
bad_sign_plain_mem_multi_bb:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function bad_sign_plain_mem_multi_bb, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:     pacda   x0, x1
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      ldr     x0, [x1]
        ldr     x0, [x1]
        cbz     x3, 1f
        autda   x0, x2
        ldr     x2, [x0]
1:
        pacda   x0, x1
        ret
        .size bad_sign_plain_mem_multi_bb, .-bad_sign_plain_mem_multi_bb

        .globl  bad_clobber_between_addr_mat_and_use_multi_bb
        .type   bad_clobber_between_addr_mat_and_use_multi_bb,@function
bad_clobber_between_addr_mat_and_use_multi_bb:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function bad_clobber_between_addr_mat_and_use_multi_bb, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:     pacda   x0, x1
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      mov     w0, w3
        adr     x0, sym
        cbz     x4, 1f
        mov     w0, w3
1:
        pacda   x0, x1
        ret
        .size bad_clobber_between_addr_mat_and_use_multi_bb, .-bad_clobber_between_addr_mat_and_use_multi_bb

        .globl  bad_clobber_between_auted_and_checked_multi_bb
        .type   bad_clobber_between_auted_and_checked_multi_bb,@function
bad_clobber_between_auted_and_checked_multi_bb:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function bad_clobber_between_auted_and_checked_multi_bb, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:     pacda   x0, x1
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      mov     w0, w3
        autda   x0, x2
        cbz     x4, 1f
        mov     w0, w3
1:
        ldr     x2, [x0]
        pacda   x0, x1
        ret
        .size bad_clobber_between_auted_and_checked_multi_bb, .-bad_clobber_between_auted_and_checked_multi_bb

        .globl  bad_clobber_between_checked_and_used_multi_bb
        .type   bad_clobber_between_checked_and_used_multi_bb,@function
bad_clobber_between_checked_and_used_multi_bb:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function bad_clobber_between_checked_and_used_multi_bb, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:     pacda   x0, x1
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      mov     w0, w3
        autda   x0, x2
        ldr     x2, [x0]
        cbz     x4, 1f
        mov     w0, w3
1:
        pacda   x0, x1
        ret
        .size bad_clobber_between_checked_and_used_multi_bb, .-bad_clobber_between_checked_and_used_multi_bb

        .globl  good_both_trusted_multi_bb
        .type   good_both_trusted_multi_bb,@function
good_both_trusted_multi_bb:
// CHECK-NOT: good_both_trusted_multi_bb
        cbz     x2, 1f
        autdb   x0, x1
        ldr     x2, [x0]
        b       2f
1:
        adr     x0, sym
2:
        pacda   x0, x1
        ret
        .size good_both_trusted_multi_bb, .-good_both_trusted_multi_bb

        .globl  bad_one_auted_one_checked_multi_bb
        .type   bad_one_auted_one_checked_multi_bb,@function
bad_one_auted_one_checked_multi_bb:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function bad_one_auted_one_checked_multi_bb, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:     pacda x0, x1
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        cbz     x2, 1f
        autdb   x0, x1
        b       2f
1:
        ldr     x3, [x0]
2:
        pacda   x0, x1
        ret
        .size bad_one_auted_one_checked_multi_bb, .-bad_one_auted_one_checked_multi_bb

// Test the detection when no CFG was reconstructed for a function.
// Test transitions between register states: none, safe-to-dereference (s-t-d), trusted:
// * trusted right away: safe address materialization
// * trusted as checked s-t-d: only check by load (FIXME: support BRK-based code sequences)
// * untrusted: s-t-d, but not checked
// * untrusted: not s-t-d, but checked
// * untrusted: not even s-t-d - from arg and from memory
// * untrusted: subreg clobbered - between address materialization and use, between auth and check, between check and use
// * untrusted: first checked then auted, auted then auted, checked then checked

        .globl  good_sign_addr_mat_nocfg
        .type   good_sign_addr_mat_nocfg,@function
good_sign_addr_mat_nocfg:
// CHECK-NOT: good_sign_addr_mat_nocfg
        adr     x3, 1f
        br      x3
1:
        adr     x0, sym
        pacda   x0, x1
        ret
        .size good_sign_addr_mat_nocfg, .-good_sign_addr_mat_nocfg

        .globl  good_sign_auted_checked_ldr_nocfg
        .type   good_sign_auted_checked_ldr_nocfg,@function
good_sign_auted_checked_ldr_nocfg:
// CHECK-NOT: good_sign_auted_checked_ldr_nocfg
        adr     x3, 1f
        br      x3
1:
        autda   x0, x2
        ldr     x2, [x0]
        pacda   x0, x1
        ret
        .size good_sign_auted_checked_ldr_nocfg, .-good_sign_auted_checked_ldr_nocfg

        .globl  bad_sign_authed_unchecked_nocfg
        .type   bad_sign_authed_unchecked_nocfg,@function
bad_sign_authed_unchecked_nocfg:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function bad_sign_authed_unchecked_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:     pacda   x0, x1
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        adr     x3, 1f
        br      x3
1:
        autda   x0, x2
        pacda   x0, x1
        ret
        .size bad_sign_authed_unchecked_nocfg, .-bad_sign_authed_unchecked_nocfg

        .globl  bad_sign_checked_not_auted_nocfg
        .type   bad_sign_checked_not_auted_nocfg,@function
bad_sign_checked_not_auted_nocfg:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function bad_sign_checked_not_auted_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:     pacda   x0, x1
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        adr     x3, 1f
        br      x3
1:
        ldr     x2, [x0]
        pacda   x0, x1
        ret
        .size bad_sign_checked_not_auted_nocfg, .-bad_sign_checked_not_auted_nocfg

        .globl  bad_sign_plain_arg_nocfg
        .type   bad_sign_plain_arg_nocfg,@function
bad_sign_plain_arg_nocfg:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function bad_sign_plain_arg_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:     pacda   x0, x1
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        adr     x3, 1f
        br      x3
1:
        pacda   x0, x1
        ret
        .size bad_sign_plain_arg_nocfg, .-bad_sign_plain_arg_nocfg

        .globl  bad_sign_plain_mem_nocfg
        .type   bad_sign_plain_mem_nocfg,@function
bad_sign_plain_mem_nocfg:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function bad_sign_plain_mem_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:     pacda   x0, x1
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      ldr     x0, [x1]
        adr     x3, 1f
        br      x3
1:
        ldr     x0, [x1]
        pacda   x0, x1
        ret
        .size bad_sign_plain_mem_nocfg, .-bad_sign_plain_mem_nocfg

        .globl  bad_clobber_between_addr_mat_and_use_nocfg
        .type   bad_clobber_between_addr_mat_and_use_nocfg,@function
bad_clobber_between_addr_mat_and_use_nocfg:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function bad_clobber_between_addr_mat_and_use_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:     pacda   x0, x1
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      mov     w0, w4
        adr     x3, 1f
        br      x3
1:
        adr     x0, sym
        mov     w0, w4
        pacda   x0, x1
        ret
        .size bad_clobber_between_addr_mat_and_use_nocfg, .-bad_clobber_between_addr_mat_and_use_nocfg

        .globl  bad_clobber_between_auted_and_checked_nocfg
        .type   bad_clobber_between_auted_and_checked_nocfg,@function
bad_clobber_between_auted_and_checked_nocfg:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function bad_clobber_between_auted_and_checked_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:     pacda   x0, x1
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      mov     w0, w4
        adr     x3, 1f
        br      x3
1:
        autda   x0, x2
        mov     w0, w4
        ldr     x2, [x0]
        pacda   x0, x1
        ret
        .size bad_clobber_between_auted_and_checked_nocfg, .-bad_clobber_between_auted_and_checked_nocfg

        .globl  bad_clobber_between_checked_and_used_nocfg
        .type   bad_clobber_between_checked_and_used_nocfg,@function
bad_clobber_between_checked_and_used_nocfg:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function bad_clobber_between_checked_and_used_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:     pacda   x0, x1
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      mov     w0, w4
        adr     x3, 1f
        br      x3
1:
        autda   x0, x2
        ldr     x2, [x0]
        mov     w0, w4
        pacda   x0, x1
        ret
        .size bad_clobber_between_checked_and_used_nocfg, .-bad_clobber_between_checked_and_used_nocfg

        .globl  bad_transition_check_then_auth_nocfg
        .type   bad_transition_check_then_auth_nocfg,@function
bad_transition_check_then_auth_nocfg:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function bad_transition_check_then_auth_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:     pacda   x0, x1
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        adr     x3, 1f
        br      x3
1:
        ldr     x2, [x0]
        autda   x0, x2
        pacda   x0, x1
        ret
        .size bad_transition_check_then_auth_nocfg, .-bad_transition_check_then_auth_nocfg

        .globl  bad_transition_auth_then_auth_nocfg
        .type   bad_transition_auth_then_auth_nocfg,@function
bad_transition_auth_then_auth_nocfg:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function bad_transition_auth_then_auth_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:     pacda   x0, x1
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        adr     x3, 1f
        br      x3
1:
        autda   x0, x2
        autda   x0, x2
        pacda   x0, x1
        ret
        .size bad_transition_auth_then_auth_nocfg, .-bad_transition_auth_then_auth_nocfg

        .globl  bad_transition_check_then_check_nocfg
        .type   bad_transition_check_then_check_nocfg,@function
bad_transition_check_then_check_nocfg:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function bad_transition_check_then_check_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:     pacda   x0, x1
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        adr     x3, 1f
        br      x3
1:
        ldr     x2, [x0]
        ldr     x2, [x0]
        pacda   x0, x1
        ret
        .size bad_transition_check_then_check_nocfg, .-bad_transition_check_then_check_nocfg

// Test resign with offset.

        .globl  good_resign_with_increment_ldr
        .type   good_resign_with_increment_ldr,@function
good_resign_with_increment_ldr:
// CHECK-NOT: good_resign_with_increment_ldr
        autda   x0, x2
        add     x0, x0, #8
        ldr     x2, [x0]
        sub     x1, x0, #16
        mov     x2, x1
        pacda   x2, x3
        ret
        .size good_resign_with_increment_ldr, .-good_resign_with_increment_ldr

        .globl  good_resign_with_increment_brk
        .type   good_resign_with_increment_brk,@function
good_resign_with_increment_brk:
// CHECK-NOT: good_resign_with_increment_brk
        autda   x0, x2
        add     x0, x0, #8
        eor     x16, x0, x0, lsl #1
        tbz     x16, #62, 1f
        brk     0xc472
1:
        mov     x2, x0
        pacda   x2, x1
        ret
        .size good_resign_with_increment_brk, .-good_resign_with_increment_brk

        .globl  bad_nonconstant_auth_increment_check
        .type   bad_nonconstant_auth_increment_check,@function
bad_nonconstant_auth_increment_check:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function bad_nonconstant_auth_increment_check, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:     pacda   x0, x1
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      add     x0, x0, x1
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   autda   x0, x2
// CHECK-NEXT:  {{[0-9a-f]+}}:   add     x0, x0, x1
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldr     x2, [x0]
// CHECK-NEXT:  {{[0-9a-f]+}}:   pacda   x0, x1
// CHECK-NEXT:  {{[0-9a-f]+}}:   ret
        autda   x0, x2
        add     x0, x0, x1
        ldr     x2, [x0]
        pacda   x0, x1
        ret
        .size bad_nonconstant_auth_increment_check, .-bad_nonconstant_auth_increment_check

        .globl  bad_nonconstant_auth_check_increment
        .type   bad_nonconstant_auth_check_increment,@function
bad_nonconstant_auth_check_increment:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function bad_nonconstant_auth_check_increment, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:     pacda   x0, x1
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      add     x0, x0, x1
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   autda   x0, x2
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldr     x2, [x0]
// CHECK-NEXT:  {{[0-9a-f]+}}:   add     x0, x0, x1
// CHECK-NEXT:  {{[0-9a-f]+}}:   pacda   x0, x1
// CHECK-NEXT:  {{[0-9a-f]+}}:   ret
        autda   x0, x2
        ldr     x2, [x0]
        add     x0, x0, x1
        pacda   x0, x1
        ret
        .size bad_nonconstant_auth_check_increment, .-bad_nonconstant_auth_check_increment

// Test that all the expected signing instructions are recornized.

        .globl  inst_pacda
        .type   inst_pacda,@function
inst_pacda:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function inst_pacda, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacda   x0, x1
        pacda   x0, x1
        ret
        .size inst_pacda, .-inst_pacda

        .globl  inst_pacdza
        .type   inst_pacdza,@function
inst_pacdza:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function inst_pacdza, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacdza  x0
        pacdza  x0
        ret
        .size inst_pacdza, .-inst_pacdza

        .globl  inst_pacdb
        .type   inst_pacdb,@function
inst_pacdb:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function inst_pacdb, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacdb   x0, x1
        pacdb   x0, x1
        ret
        .size inst_pacdb, .-inst_pacdb

        .globl  inst_pacdzb
        .type   inst_pacdzb,@function
inst_pacdzb:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function inst_pacdzb, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacdzb  x0
        pacdzb  x0
        ret
        .size inst_pacdzb, .-inst_pacdzb

        .globl  inst_pacia
        .type   inst_pacia,@function
inst_pacia:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function inst_pacia, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacia   x0, x1
        pacia   x0, x1
        ret
        .size inst_pacia, .-inst_pacia

        .globl  inst_pacia1716
        .type   inst_pacia1716,@function
inst_pacia1716:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function inst_pacia1716, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacia1716
        pacia1716
        ret
        .size inst_pacia1716, .-inst_pacia1716

        .globl  inst_paciasp
        .type   inst_paciasp,@function
inst_paciasp:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function inst_paciasp, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      paciasp
        mov     x30, x0
        paciasp          // signs LR
        autiasp
        ret
        .size inst_paciasp, .-inst_paciasp

        .globl  inst_paciaz
        .type   inst_paciaz,@function
inst_paciaz:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function inst_paciaz, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      paciaz
        mov     x30, x0
        paciaz           // signs LR
        autiaz
        ret
        .size inst_paciaz, .-inst_paciaz

        .globl  inst_paciza
        .type   inst_paciza,@function
inst_paciza:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function inst_paciza, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      paciza  x0
        paciza  x0
        ret
        .size inst_paciza, .-inst_paciza

        .globl  inst_pacia171615
        .type   inst_pacia171615,@function
inst_pacia171615:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function inst_pacia171615, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacia171615
        mov     x30, x0
        pacia171615      // signs LR
        autia171615
        ret
        .size inst_pacia171615, .-inst_pacia171615

        .globl  inst_paciasppc
        .type   inst_paciasppc,@function
inst_paciasppc:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function inst_paciasppc, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      paciasppc
        mov     x30, x0
1:
        paciasppc        // signs LR
        autiasppc  1b
        ret
        .size inst_paciasppc, .-inst_paciasppc

        .globl  inst_pacib
        .type   inst_pacib,@function
inst_pacib:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function inst_pacib, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacib   x0, x1
        pacib   x0, x1
        ret
        .size inst_pacib, .-inst_pacib

        .globl  inst_pacib1716
        .type   inst_pacib1716,@function
inst_pacib1716:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function inst_pacib1716, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacib1716
        pacib1716
        ret
        .size inst_pacib1716, .-inst_pacib1716

        .globl  inst_pacibsp
        .type   inst_pacibsp,@function
inst_pacibsp:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function inst_pacibsp, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacibsp
        mov     x30, x0
        pacibsp          // signs LR
        autibsp
        ret
        .size inst_pacibsp, .-inst_pacibsp

        .globl  inst_pacibz
        .type   inst_pacibz,@function
inst_pacibz:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function inst_pacibz, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacibz
        mov     x30, x0
        pacibz           // signs LR
        autibz
        ret
        .size inst_pacibz, .-inst_pacibz

        .globl  inst_pacizb
        .type   inst_pacizb,@function
inst_pacizb:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function inst_pacizb, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacizb  x0
        pacizb  x0
        ret
        .size inst_pacizb, .-inst_pacizb

        .globl  inst_pacib171615
        .type   inst_pacib171615,@function
inst_pacib171615:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function inst_pacib171615, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacib171615
        mov     x30, x0
        pacib171615      // signs LR
        autib171615
        ret
        .size inst_pacib171615, .-inst_pacib171615

        .globl  inst_pacibsppc
        .type   inst_pacibsppc,@function
inst_pacibsppc:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function inst_pacibsppc, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacibsppc
        mov     x30, x0
1:
        pacibsppc        // signs LR
        autibsppc  1b
        ret
        .size inst_pacibsppc, .-inst_pacibsppc

        .globl  inst_pacnbiasppc
        .type   inst_pacnbiasppc,@function
inst_pacnbiasppc:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function inst_pacnbiasppc, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacnbiasppc
        mov     x30, x0
1:
        pacnbiasppc      // signs LR
        autiasppc  1b
        ret
        .size inst_pacnbiasppc, .-inst_pacnbiasppc

        .globl  inst_pacnbibsppc
        .type   inst_pacnbibsppc,@function
inst_pacnbibsppc:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function inst_pacnbibsppc, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacnbibsppc
        mov     x30, x0
1:
        pacnbibsppc      // signs LR
        autibsppc  1b
        ret
        .size inst_pacnbibsppc, .-inst_pacnbibsppc

// Test that write-back forms of LDRA(A|B) instructions are handled properly.

        .globl  inst_ldraa_wb
        .type   inst_ldraa_wb,@function
inst_ldraa_wb:
// CHECK-NOT: inst_ldraa_wb
        ldraa   x2, [x0]!
        pacda   x0, x1
        ret
        .size inst_ldraa_wb, .-inst_ldraa_wb

        .globl  inst_ldrab_wb
        .type   inst_ldrab_wb,@function
inst_ldrab_wb:
// CHECK-NOT: inst_ldrab_wb
        ldrab   x2, [x0]!
        pacda   x0, x1
        ret
        .size inst_ldrab_wb, .-inst_ldrab_wb

// Non write-back forms of LDRA(A|B) instructions do not modify the address
// register, and thus do not make it safe.

        .globl  inst_ldraa_non_wb
        .type   inst_ldraa_non_wb,@function
inst_ldraa_non_wb:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function inst_ldraa_non_wb, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacdb   x0, x1
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        ldraa   x2, [x0]
        pacdb   x0, x1
        ret
        .size inst_ldraa_non_wb, .-inst_ldraa_non_wb

        .globl  inst_ldrab_non_wb
        .type   inst_ldrab_non_wb,@function
inst_ldrab_non_wb:
// CHECK-LABEL: GS-PAUTH: signing oracle found in function inst_ldrab_non_wb, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      pacda   x0, x1
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        ldrab   x2, [x0]
        pacda   x0, x1
        ret
        .size inst_ldrab_non_wb, .-inst_ldrab_non_wb

        .globl  main
        .type   main,@function
main:
        mov     x0, 0
        ret
        .size   main, .-main
