// RUN: %clang %cflags -march=armv9.5-a+pauth-lr -mbranch-protection=pac-ret %s %p/../../Inputs/asm_main.c -o %t.exe
// RUN: llvm-bolt-binary-analysis --scanners=pacret %t.exe 2>&1 | FileCheck %s

        .text

        .globl  f1
        .type   f1,@function
f1:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        // autiasp
// CHECK-LABEL: GS-PAUTH: non-protected ret found in function f1, basic block {{[0-9a-zA-Z.]+}}, at address
// CHECK-NEXT:    The instruction is     {{[0-9a-f]+}}:       ret
// CHECK-NEXT:    The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:    1. {{[0-9a-f]+}}: ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT: {{[0-9a-f]+}}:   add     x0, x0, #0x3
// CHECK-NEXT: {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT: {{[0-9a-f]+}}:   ret
        ret
        .size f1, .-f1


        .globl  f_intermediate_overwrite1
        .type   f_intermediate_overwrite1,@function
f_intermediate_overwrite1:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        autiasp
        ldp     x29, x30, [sp], #16
// CHECK-LABEL: GS-PAUTH: non-protected ret found in function f_intermediate_overwrite1, basic block {{[0-9a-zA-Z.]+}}
// CHECK-NEXT:    The instruction is     {{[0-9a-f]+}}:       ret
// CHECK-NEXT:    The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:    1. {{[0-9a-f]+}}: ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT: {{[0-9a-f]+}}:   add     x0, x0, #0x3
// CHECK-NEXT: {{[0-9a-f]+}}:   autiasp
// CHECK-NEXT: {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT: {{[0-9a-f]+}}:   ret
        ret
        .size f_intermediate_overwrite1, .-f_intermediate_overwrite1

        .globl  f_intermediate_overwrite2
        .type   f_intermediate_overwrite2,@function
f_intermediate_overwrite2:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autiasp
        mov     x30, x0
// CHECK-LABEL: GS-PAUTH: non-protected ret found in function f_intermediate_overwrite2, basic block {{[0-9a-zA-Z.]+}}, at address
// CHECK-NEXT:    The instruction is     {{[0-9a-f]+}}:       ret
// CHECK-NEXT:    The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:    1. {{[0-9a-f]+}}: mov     x30, x0
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT: {{[0-9a-f]+}}:   add     x0, x0, #0x3
// CHECK-NEXT: {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT: {{[0-9a-f]+}}:   autiasp
// CHECK-NEXT: {{[0-9a-f]+}}:   mov     x30, x0
// CHECK-NEXT: {{[0-9a-f]+}}:   ret
        ret
        .size f_intermediate_overwrite2, .-f_intermediate_overwrite2

        .globl  f_intermediate_read
        .type   f_intermediate_read,@function
f_intermediate_read:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autiasp
        mov     x0, x30
// CHECK-NOT: function f_intermediate_read
        ret
        .size f_intermediate_read, .-f_intermediate_read

        .globl  f_intermediate_overwrite3
        .type   f_intermediate_overwrite3,@function
f_intermediate_overwrite3:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autiasp
        mov     w30, w0
// CHECK-LABEL: GS-PAUTH: non-protected ret found in function f_intermediate_overwrite3, basic block {{[0-9a-zA-Z.]+}}, at address
// CHECK-NEXT:    The instruction is     {{[0-9a-f]+}}:       ret
// CHECK-NEXT:    The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:    1. {{[0-9a-f]+}}: mov     w30, w0
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT: {{[0-9a-f]+}}:   add     x0, x0, #0x3
// CHECK-NEXT: {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT: {{[0-9a-f]+}}:   autiasp
// CHECK-NEXT: {{[0-9a-f]+}}:   mov     w30, w0
// CHECK-NEXT: {{[0-9a-f]+}}:   ret
        ret
        .size f_intermediate_overwrite3, .-f_intermediate_overwrite3

        .globl  f_nonx30_ret
        .type   f_nonx30_ret,@function
f_nonx30_ret:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        mov     x16, x30
        autiasp
// CHECK-LABEL: GS-PAUTH: non-protected ret found in function f_nonx30_ret, basic block {{[0-9a-zA-Z.]+}}, at address
// CHECK-NEXT:    The instruction is     {{[0-9a-f]+}}:       ret     x16
// CHECK-NEXT:    The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:    1. {{[0-9a-f]+}}: mov     x16, x30
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT: {{[0-9a-f]+}}:   add     x0, x0, #0x3
// CHECK-NEXT: {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT: {{[0-9a-f]+}}:   mov     x16, x30
// CHECK-NEXT: {{[0-9a-f]+}}:   autiasp
// CHECK-NEXT: {{[0-9a-f]+}}:   ret     x16
        ret     x16
        .size f_nonx30_ret, .-f_nonx30_ret

        .globl  f_nonx30_ret_ok
        .type   f_nonx30_ret_ok,@function
f_nonx30_ret_ok:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        ldp     x29, x30, [sp], #16
        autiasp
        mov     x16, x30
// CHECK-NOT: function f_nonx30_ret_ok
        ret     x16
        .size f_nonx30_ret_ok, .-f_nonx30_ret_ok

        .globl  f_detect_clobbered_x30_passed_to_other
        .type   f_detect_clobbered_x30_passed_to_other,@function
f_detect_clobbered_x30_passed_to_other:
        str x30, [sp]
        ldr x30, [sp]
// FIXME: Ideally, the pac-ret scanner would report on the following instruction, which
// performs a tail call, that x30 might be attacker-controlled.
// CHECK-NOT: function f_detect_clobbered_x30_passed_to_other
        b   f_tail_called
        .size f_detect_clobbered_x30_passed_to_other, .-f_detect_clobbered_x30_passed_to_other

        .globl  f_tail_called
        .type   f_tail_called,@function
f_tail_called:
        ret
        .size f_tail_called, .-f_tail_called

        .globl  f_nonx30_ret_non_auted
        .type   f_nonx30_ret_non_auted,@function
f_nonx30_ret_non_auted:
// x1 is neither authenticated nor implicitly considered safe at function entry.
// Note that we assume it's fine for x30 to not be authenticated before
// returning to, as assuming that x30 is not attacker controlled at function
// entry is part (implicitly) of the pac-ret hardening scheme.
//
// CHECK-LABEL: GS-PAUTH: non-protected ret found in function f_nonx30_ret_non_auted, basic block {{[0-9a-zA-Z.]+}}, at address
// CHECK-NEXT:    The instruction is     {{[0-9a-f]+}}:       ret
// CHECK-NEXT:    The 0 instructions that write to the affected registers after any authentication are:
        ret     x1
        .size f_nonx30_ret_non_auted, .-f_nonx30_ret_non_auted


        .globl  f_callclobbered_x30
        .type   f_callclobbered_x30,@function
f_callclobbered_x30:
        bl      g
// CHECK-LABEL: GS-PAUTH: non-protected ret found in function f_callclobbered_x30, basic block {{[0-9a-zA-Z.]+}}, at address
// CHECK-NEXT:    The instruction is     {{[0-9a-f]+}}:       ret
// CHECK-NEXT:    The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:    1. {{[0-9a-f]+}}: bl
        ret
        .size f_callclobbered_x30, .-f_callclobbered_x30

        .globl  f_callclobbered_calleesaved
        .type   f_callclobbered_calleesaved,@function
f_callclobbered_calleesaved:
        bl      g
// CHECK-LABEL: GS-PAUTH: non-protected ret found in function f_callclobbered_calleesaved, basic block {{[0-9a-zA-Z.]+}}, at address
// CHECK-NEXT:    The instruction is     {{[0-9a-f]+}}:       ret x19
// CHECK-NEXT:    The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:    1. {{[0-9a-f]+}}: bl
        // x19, according to the Arm ABI (AAPCS) is a callee-saved register.
        // Therefore, if function g respects the AAPCS, it should not write
        // anything to x19. However, we can't know whether function g actually
        // does respect the AAPCS rules, so the scanner should assume x19 can
        // get overwritten, and report a gadget if the code does not properly
        // deal with that.
        // Furthermore, there's a good chance that callee-saved registers have
        // been saved on the stack at some point during execution of the callee,
        // and so should be considered as potentially modified by an
        // attacker/written to.
        ret x19
        .size f_callclobbered_calleesaved, .-f_callclobbered_calleesaved

        .globl  f_unreachable_instruction
        .type   f_unreachable_instruction,@function
f_unreachable_instruction:
// CHECK-LABEL: GS-PAUTH: Warning: unreachable instruction found in function f_unreachable_instruction, basic block {{[0-9a-zA-Z.]+}}, at address
// CHECK-NEXT:    The instruction is     {{[0-9a-f]+}}:       add     x0, x1, x2
// CHECK-NOT:   instructions that write to the affected registers after any authentication are:
        b       1f
        add     x0, x1, x2
1:
        ret
        .size f_unreachable_instruction, .-f_unreachable_instruction

// Expected false positive: without CFG, the state is reset to all-unsafe
// after an unconditional branch.

        .globl  state_is_reset_after_indirect_branch_nocfg
        .type   state_is_reset_after_indirect_branch_nocfg,@function
state_is_reset_after_indirect_branch_nocfg:
// CHECK-LABEL: GS-PAUTH: non-protected ret found in function state_is_reset_after_indirect_branch_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:         ret
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        adr     x2, 1f
        br      x2
1:
        ret
        .size state_is_reset_after_indirect_branch_nocfg, .-state_is_reset_after_indirect_branch_nocfg

/// Now do a basic sanity check on every different Authentication instruction:

        .globl  f_autiasp
        .type   f_autiasp,@function
f_autiasp:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autiasp
// CHECK-NOT: function f_autiasp
        ret
        .size f_autiasp, .-f_autiasp

        .globl  f_autibsp
        .type   f_autibsp,@function
f_autibsp:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autibsp
// CHECK-NOT: function f_autibsp
        ret
        .size f_autibsp, .-f_autibsp

        .globl  f_autiaz
        .type   f_autiaz,@function
f_autiaz:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autiaz
// CHECK-NOT: function f_autiaz
        ret
        .size f_autiaz, .-f_autiaz

        .globl  f_autibz
        .type   f_autibz,@function
f_autibz:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autibz
// CHECK-NOT: function f_autibz
        ret
        .size f_autibz, .-f_autibz

        .globl  f_autia1716
        .type   f_autia1716,@function
f_autia1716:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autia1716
// CHECK-LABEL: GS-PAUTH: non-protected ret found in function f_autia1716, basic block {{[0-9a-zA-Z.]+}}, at address
// CHECK-NEXT:    The instruction is     {{[0-9a-f]+}}:       ret
// CHECK-NEXT:    The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:    1. {{[0-9a-f]+}}: ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT: {{[0-9a-f]+}}:   add     x0, x0, #0x3
// CHECK-NEXT: {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT: {{[0-9a-f]+}}:   autia1716
// CHECK-NEXT: {{[0-9a-f]+}}:   ret
        ret
        .size f_autia1716, .-f_autia1716

        .globl  f_autib1716
        .type   f_autib1716,@function
f_autib1716:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autib1716
// CHECK-LABEL: GS-PAUTH: non-protected ret found in function f_autib1716, basic block {{[0-9a-zA-Z.]+}}, at address
// CHECK-NEXT:    The instruction is     {{[0-9a-f]+}}:       ret
// CHECK-NEXT:    The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:    1. {{[0-9a-f]+}}: ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT: {{[0-9a-f]+}}:   add     x0, x0, #0x3
// CHECK-NEXT: {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT: {{[0-9a-f]+}}:   autib1716
// CHECK-NEXT: {{[0-9a-f]+}}:   ret
        ret
        .size f_autib1716, .-f_autib1716

        .globl  f_autiax12
        .type   f_autiax12,@function
f_autiax12:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autia   x12, sp
// CHECK-LABEL: GS-PAUTH: non-protected ret found in function f_autiax12, basic block {{[0-9a-zA-Z.]+}}, at address
// CHECK-NEXT:    The instruction is     {{[0-9a-f]+}}:       ret
// CHECK-NEXT:    The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:    1. {{[0-9a-f]+}}: ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT: {{[0-9a-f]+}}:   add     x0, x0, #0x3
// CHECK-NEXT: {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT: {{[0-9a-f]+}}:   autia   x12, sp
// CHECK-NEXT: {{[0-9a-f]+}}:   ret
        ret
        .size f_autiax12, .-f_autiax12

        .globl  f_autibx12
        .type   f_autibx12,@function
f_autibx12:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autib   x12, sp
// CHECK-LABEL: GS-PAUTH: non-protected ret found in function f_autibx12, basic block {{[0-9a-zA-Z.]+}}, at address
// CHECK-NEXT:    The instruction is     {{[0-9a-f]+}}:       ret
// CHECK-NEXT:    The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:    1. {{[0-9a-f]+}}: ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT: {{[0-9a-f]+}}:   add     x0, x0, #0x3
// CHECK-NEXT: {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT: {{[0-9a-f]+}}:   autib   x12, sp
// CHECK-NEXT: {{[0-9a-f]+}}:   ret
        ret
        .size f_autibx12, .-f_autibx12

        .globl  f_autiax30
        .type   f_autiax30,@function
f_autiax30:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autia   x30, sp
// CHECK-NOT: function f_autiax30
        ret
        .size f_autiax30, .-f_autiax30

        .globl  f_autibx30
        .type   f_autibx30,@function
f_autibx30:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autib   x30, sp
// CHECK-NOT: function f_autibx30
        ret
        .size f_autibx30, .-f_autibx30


        .globl  f_autdax12
        .type   f_autdax12,@function
f_autdax12:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autda   x12, sp
// CHECK-LABEL: GS-PAUTH: non-protected ret found in function f_autdax12, basic block {{[0-9a-zA-Z.]+}}, at address
// CHECK-NEXT:    The instruction is     {{[0-9a-f]+}}:       ret
// CHECK-NEXT:    The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:    1. {{[0-9a-f]+}}: ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT: {{[0-9a-f]+}}:   add     x0, x0, #0x3
// CHECK-NEXT: {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT: {{[0-9a-f]+}}:   autda   x12, sp
// CHECK-NEXT: {{[0-9a-f]+}}:   ret
        ret
        .size f_autdax12, .-f_autdax12

        .globl  f_autdbx12
        .type   f_autdbx12,@function
f_autdbx12:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autdb   x12, sp
// CHECK-LABEL: GS-PAUTH: non-protected ret found in function f_autdbx12, basic block {{[0-9a-zA-Z.]+}}, at address
// CHECK-NEXT:    The instruction is     {{[0-9a-f]+}}:       ret
// CHECK-NEXT:    The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:    1. {{[0-9a-f]+}}: ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT: {{[0-9a-f]+}}:   add     x0, x0, #0x3
// CHECK-NEXT: {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT: {{[0-9a-f]+}}:   autdb   x12, sp
// CHECK-NEXT: {{[0-9a-f]+}}:   ret
        ret
        .size f_autdbx12, .-f_autdbx12

        .globl  f_autdax30
        .type   f_autdax30,@function
f_autdax30:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autda   x30, sp
// CHECK-NOT: function f_autdax30
        ret
        .size f_autdax30, .-f_autdax30

        .globl  f_autdbx30
        .type   f_autdbx30,@function
f_autdbx30:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autdb   x30, sp
// CHECK-NOT: function f_autdbx30
        ret
        .size f_autdbx30, .-f_autdbx30


        .globl  f_autizax12
        .type   f_autizax12,@function
f_autizax12:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autiza  x12
// CHECK-LABEL: GS-PAUTH: non-protected ret found in function f_autizax12, basic block {{[0-9a-zA-Z.]+}}, at address
// CHECK-NEXT:    The instruction is     {{[0-9a-f]+}}:       ret
// CHECK-NEXT:    The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:    1. {{[0-9a-f]+}}: ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT: {{[0-9a-f]+}}:   add     x0, x0, #0x3
// CHECK-NEXT: {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT: {{[0-9a-f]+}}:   autiza  x12
// CHECK-NEXT: {{[0-9a-f]+}}:   ret
        ret
        .size f_autizax12, .-f_autizax12

        .globl  f_autizbx12
        .type   f_autizbx12,@function
f_autizbx12:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autizb  x12
// CHECK-LABEL: GS-PAUTH: non-protected ret found in function f_autizbx12, basic block {{[0-9a-zA-Z.]+}}, at address
// CHECK-NEXT:    The instruction is     {{[0-9a-f]+}}:       ret
// CHECK-NEXT:    The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:    1. {{[0-9a-f]+}}: ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT: {{[0-9a-f]+}}:   add     x0, x0, #0x3
// CHECK-NEXT: {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT: {{[0-9a-f]+}}:   autizb  x12
// CHECK-NEXT: {{[0-9a-f]+}}:   ret
        ret
        .size f_autizbx12, .-f_autizbx12

        .globl  f_autizax30
        .type   f_autizax30,@function
f_autizax30:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autiza  x30
// CHECK-NOT: function f_autizax30
        ret
        .size f_autizax30, .-f_autizax30

        .globl  f_autizbx30
        .type   f_autizbx30,@function
f_autizbx30:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autizb  x30
// CHECK-NOT: function f_autizbx30
        ret
        .size f_autizbx30, .-f_autizbx30


        .globl  f_autdzax12
        .type   f_autdzax12,@function
f_autdzax12:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autdza  x12
// CHECK-LABEL: GS-PAUTH: non-protected ret found in function f_autdzax12, basic block {{[0-9a-zA-Z.]+}}, at address
// CHECK-NEXT:    The instruction is     {{[0-9a-f]+}}:       ret
// CHECK-NEXT:    The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:    1. {{[0-9a-f]+}}: ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT: {{[0-9a-f]+}}:   add     x0, x0, #0x3
// CHECK-NEXT: {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT: {{[0-9a-f]+}}:   autdza  x12
// CHECK-NEXT: {{[0-9a-f]+}}:   ret
        ret
        .size f_autdzax12, .-f_autdzax12

        .globl  f_autdzbx12
        .type   f_autdzbx12,@function
f_autdzbx12:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autdzb  x12
// CHECK-LABEL: GS-PAUTH: non-protected ret found in function f_autdzbx12, basic block {{[0-9a-zA-Z.]+}}, at address
// CHECK-NEXT:    The instruction is     {{[0-9a-f]+}}:       ret
// CHECK-NEXT:    The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:    1. {{[0-9a-f]+}}: ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT: {{[0-9a-f]+}}:   add     x0, x0, #0x3
// CHECK-NEXT: {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT: {{[0-9a-f]+}}:   autdzb  x12
// CHECK-NEXT: {{[0-9a-f]+}}:   ret
        ret
        .size f_autdzbx12, .-f_autdzbx12

        .globl  f_autdzax30
        .type   f_autdzax30,@function
f_autdzax30:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autdza  x30
// CHECK-NOT: function f_autdzax30
        ret
        .size f_autdzax30, .-f_autdzax30

        .globl  f_autdzbx30
        .type   f_autdzbx30,@function
f_autdzbx30:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autdzb  x30
// CHECK-NOT: function f_autdzbx30
        ret
        .size f_autdzbx30, .-f_autdzbx30

        .globl  f_retaa
        .type   f_retaa,@function
f_retaa:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
// CHECK-NOT: function f_retaa
        retaa
        .size f_retaa, .-f_retaa

        .globl  f_retab
        .type   f_retab,@function
f_retab:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
// CHECK-NOT: function f_retab
        retab
        .size f_retab, .-f_retab

        .globl  f_eretaa
        .type   f_eretaa,@function
f_eretaa:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
// CHECK-LABEL: GS-PAUTH: Warning: pac-ret analysis could not analyze this return instruction in function f_eretaa, basic block {{[0-9a-zA-Z.]+}}, at address
// CHECK-NEXT:   The instruction is     {{[0-9a-f]+}}:       eretaa
        eretaa
        .size f_eretaa, .-f_eretaa

        .globl  f_eretab
        .type   f_eretab,@function
f_eretab:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
// CHECK-LABEL: GS-PAUTH: Warning: pac-ret analysis could not analyze this return instruction in function f_eretab, basic block {{[0-9a-zA-Z.]+}}, at address
// CHECK-NEXT:   The instruction is     {{[0-9a-f]+}}:       eretab
        eretab
        .size f_eretab, .-f_eretab

        .globl  f_eret
        .type   f_eret,@function
f_eret:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
// CHECK-LABEL: GS-PAUTH: Warning: pac-ret analysis could not analyze this return instruction in function f_eret, basic block {{[0-9a-zA-Z.]+}}, at address
// CHECK-NEXT:   The instruction is     {{[0-9a-f]+}}:       eret
        eret
        .size f_eret, .-f_eret

        .globl f_movx30reg
        .type   f_movx30reg,@function
f_movx30reg:
// CHECK-LABEL: GS-PAUTH: non-protected ret found in function f_movx30reg, basic block {{[0-9a-zA-Z.]+}}, at address
// CHECK-NEXT:    The instruction is     {{[0-9a-f]+}}:       ret
// CHECK-NEXT:    The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:    1. {{[0-9a-f]+}}: mov x30, x22
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT: {{[0-9a-f]+}}:   mov     x30, x22
// CHECK-NEXT: {{[0-9a-f]+}}:   ret
        mov     x30, x22
        ret
        .size f_movx30reg, .-f_movx30reg

        .globl  f_autiasppci
        .type   f_autiasppci,@function
f_autiasppci:
0:
        pacnbiasppc
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autiasppc 0b
// CHECK-NOT: function f_autiasppci
        ret
        .size f_autiasppci, .-f_autiasppci

        .globl  f_autibsppci
        .type   f_autibsppci,@function
f_autibsppci:
0:
        pacnbibsppc
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autibsppc 0b
// CHECK-NOT: function f_autibsppci
        ret
        .size f_autibsppci, .-f_autibsppci

        .globl  f_autiasppcr
        .type   f_autiasppcr,@function

f_autiasppcr:
0:
        pacnbiasppc
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        adr     x28, 0b
        autiasppcr x28
// CHECK-NOT: function f_autiasppcr
        ret
        .size f_autiasppcr, .-f_autiasppcr

f_autibsppcr:
0:
        pacnbibsppc
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        adr     x28, 0b
        autibsppcr x28
// CHECK-NOT: function f_autibsppcr
        ret
        .size f_autibsppcr, .-f_autibsppcr

        .globl  f_retaasppci
        .type   f_retaasppci,@function
f_retaasppci:
0:
        pacnbiasppc
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
// CHECK-NOT: function f_retaasppci
        retaasppc 0b
        .size f_retaasppci, .-f_retaasppci

        .globl  f_retabsppci
        .type   f_retabsppci,@function
f_retabsppci:
0:
        pacnbibsppc
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
// CHECK-NOT: function f_retabsppci
        retabsppc 0b
        .size f_retabsppci, .-f_retabsppci

        .globl  f_retaasppcr
        .type   f_retaasppcr,@function

f_retaasppcr:
0:
        pacnbiasppc
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        adr     x28, 0b
// CHECK-NOT: function f_retaasppcr
        retaasppcr x28
        .size f_retaasppcr, .-f_retaasppcr

f_retabsppcr:
0:
        pacnbibsppc
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        adr     x28, 0b
// CHECK-NOT: function f_retabsppcr
        retabsppcr x28
        .size f_retabsppcr, .-f_retabsppcr

        .globl  f_autia171615
        .type   f_autia171615,@function
f_autia171615:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autia171615
// CHECK-LABEL: GS-PAUTH: non-protected ret found in function f_autia171615, basic block {{[0-9a-zA-Z.]+}}, at address
// CHECK-NEXT:    The instruction is     {{[0-9a-f]+}}:       ret
// CHECK-NEXT:    The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:    1. {{[0-9a-f]+}}: ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT: {{[0-9a-f]+}}:   add     x0, x0, #0x3
// CHECK-NEXT: {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT: {{[0-9a-f]+}}:   autia171615
// CHECK-NEXT: {{[0-9a-f]+}}:   ret
        ret
        .size f_autia171615, .-f_autia171615

        .globl  f_autib171615
        .type   f_autib171615,@function
f_autib171615:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autib171615
// CHECK-LABEL: GS-PAUTH: non-protected ret found in function f_autib171615, basic block {{[0-9a-zA-Z.]+}}, at address
// CHECK-NEXT:    The instruction is     {{[0-9a-f]+}}:       ret
// CHECK-NEXT:    The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:    1. {{[0-9a-f]+}}: ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT: {{[0-9a-f]+}}:   add     x0, x0, #0x3
// CHECK-NEXT: {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT: {{[0-9a-f]+}}:   autib171615
// CHECK-NEXT: {{[0-9a-f]+}}:   ret
        ret
        .size f_autib171615, .-f_autib171615

