// RUN: %clang %cflags -march=armv8.3-a -mbranch-protection=pac-ret %s %p/../../Inputs/asm_main.c -o %t.exe
// RUN: llvm-bolt-binary-analysis --scanners=pacret %t.exe 2>&1 | FileCheck %s

        .text

        .globl  f1
        .type   f1,@function
f1:
        hint    #25
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        // autiasp
// CHECK-LABEL: GS-PACRET: non-protected ret found in function f1, basic block .LBB{{[0-9]+}}, at address
// CHECK-NEXT:    The return instruction is     {{[0-9a-f]+}}:       ret
// CHECK-NEXT:    The 1 instructions that write to the return register after any authentication are:
// CHECK-NEXT:    1. {{[0-9a-f]+}}: ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT: {{[0-9a-f]+}}:   paciasp
// CHECK-NEXT: {{[0-9a-f]+}}:   stp     x29, x30, [sp, #-0x10]!
// CHECK-NEXT: {{[0-9a-f]+}}:   mov     x29, sp
// CHECK-NEXT: {{[0-9a-f]+}}:   bl      g@PLT
// CHECK-NEXT: {{[0-9a-f]+}}:   add     x0, x0, #0x3
// CHECK-NEXT: {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT: {{[0-9a-f]+}}:   ret
        ret
        .size f1, .-f1


        .globl  f_intermediate_overwrite1
        .type   f_intermediate_overwrite1,@function
f_intermediate_overwrite1:
        hint    #25
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        autiasp
        ldp     x29, x30, [sp], #16
// CHECK-LABEL: GS-PACRET: non-protected ret found in function f_intermediate_overwrite1, basic block .LBB
// CHECK-NEXT:    The return instruction is     {{[0-9a-f]+}}:       ret
// CHECK-NEXT:    The 1 instructions that write to the return register after any authentication are:
// CHECK-NEXT:    1. {{[0-9a-f]+}}: ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT: {{[0-9a-f]+}}:   paciasp
// CHECK-NEXT: {{[0-9a-f]+}}:   stp     x29, x30, [sp, #-0x10]!
// CHECK-NEXT: {{[0-9a-f]+}}:   mov     x29, sp
// CHECK-NEXT: {{[0-9a-f]+}}:   bl      g@PLT
// CHECK-NEXT: {{[0-9a-f]+}}:   add     x0, x0, #0x3
// CHECK-NEXT: {{[0-9a-f]+}}:   autiasp
// CHECK-NEXT: {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT: {{[0-9a-f]+}}:   ret
        ret
        .size f_intermediate_overwrite1, .-f_intermediate_overwrite1

        .globl  f_intermediate_overwrite2
        .type   f_intermediate_overwrite2,@function
f_intermediate_overwrite2:
        hint    #25
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autiasp
        mov     x30, x0
// CHECK-LABEL: GS-PACRET: non-protected ret found in function f_intermediate_overwrite2, basic block .LBB{{[0-9]+}}, at address
// CHECK-NEXT:    The return instruction is     {{[0-9a-f]+}}:       ret
// CHECK-NEXT:    The 1 instructions that write to the return register after any authentication are:
// CHECK-NEXT:    1. {{[0-9a-f]+}}: mov     x30, x0
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT: {{[0-9a-f]+}}:   paciasp
// CHECK-NEXT: {{[0-9a-f]+}}:   stp     x29, x30, [sp, #-0x10]!
// CHECK-NEXT: {{[0-9a-f]+}}:   mov     x29, sp
// CHECK-NEXT: {{[0-9a-f]+}}:   bl      g@PLT
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
        hint    #25
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
        hint    #25
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autiasp
        mov     w30, w0
// CHECK-LABEL: GS-PACRET: non-protected ret found in function f_intermediate_overwrite3, basic block .LBB{{[0-9]+}}, at address
// CHECK-NEXT:    The return instruction is     {{[0-9a-f]+}}:       ret
// CHECK-NEXT:    The 1 instructions that write to the return register after any authentication are:
// CHECK-NEXT:    1. {{[0-9a-f]+}}: mov     w30, w0
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT: {{[0-9a-f]+}}:   paciasp
// CHECK-NEXT: {{[0-9a-f]+}}:   stp     x29, x30, [sp, #-0x10]!
// CHECK-NEXT: {{[0-9a-f]+}}:   mov     x29, sp
// CHECK-NEXT: {{[0-9a-f]+}}:   bl      g@PLT
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
        hint    #25
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        mov     x16, x30
        autiasp
// CHECK-LABEL: GS-PACRET: non-protected ret found in function f_nonx30_ret, basic block .LBB{{[0-9]+}}, at address
// CHECK-NEXT:    The return instruction is     {{[0-9a-f]+}}:       ret     x16
// CHECK-NEXT:    The 1 instructions that write to the return register after any authentication are:
// CHECK-NEXT:    1. {{[0-9a-f]+}}: mov     x16, x30
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT: {{[0-9a-f]+}}:   paciasp
// CHECK-NEXT: {{[0-9a-f]+}}:   stp     x29, x30, [sp, #-0x10]!
// CHECK-NEXT: {{[0-9a-f]+}}:   mov     x29, sp
// CHECK-NEXT: {{[0-9a-f]+}}:   bl      g@PLT
// CHECK-NEXT: {{[0-9a-f]+}}:   add     x0, x0, #0x3
// CHECK-NEXT: {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT: {{[0-9a-f]+}}:   mov     x16, x30
// CHECK-NEXT: {{[0-9a-f]+}}:   autiasp
// CHECK-NEXT: {{[0-9a-f]+}}:   ret     x16
        ret     x16
        .size f_nonx30_ret, .-f_nonx30_ret


/// Now do a basic sanity check on every different Authentication instruction:

        .globl  f_autiasp
        .type   f_autiasp,@function
f_autiasp:
        hint    #25
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
        hint    #25
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
        hint    #25
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
        hint    #25
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
        hint    #25
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autia1716
// CHECK-LABEL: GS-PACRET: non-protected ret found in function f_autia1716, basic block .LBB{{[0-9]+}}, at address
// CHECK-NEXT:    The return instruction is     {{[0-9a-f]+}}:       ret
// CHECK-NEXT:    The 1 instructions that write to the return register after any authentication are:
// CHECK-NEXT:    1. {{[0-9a-f]+}}: ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT: {{[0-9a-f]+}}:   paciasp
// CHECK-NEXT: {{[0-9a-f]+}}:   stp     x29, x30, [sp, #-0x10]!
// CHECK-NEXT: {{[0-9a-f]+}}:   mov     x29, sp
// CHECK-NEXT: {{[0-9a-f]+}}:   bl      g@PLT
// CHECK-NEXT: {{[0-9a-f]+}}:   add     x0, x0, #0x3
// CHECK-NEXT: {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT: {{[0-9a-f]+}}:   autia1716
// CHECK-NEXT: {{[0-9a-f]+}}:   ret
        ret
        .size f_autia1716, .-f_autia1716

        .globl  f_autib1716
        .type   f_autib1716,@function
f_autib1716:
        hint    #25
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autib1716
// CHECK-LABEL: GS-PACRET: non-protected ret found in function f_autib1716, basic block .LBB{{[0-9]+}}, at address
// CHECK-NEXT:    The return instruction is     {{[0-9a-f]+}}:       ret
// CHECK-NEXT:    The 1 instructions that write to the return register after any authentication are:
// CHECK-NEXT:    1. {{[0-9a-f]+}}: ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT: {{[0-9a-f]+}}:   paciasp
// CHECK-NEXT: {{[0-9a-f]+}}:   stp     x29, x30, [sp, #-0x10]!
// CHECK-NEXT: {{[0-9a-f]+}}:   mov     x29, sp
// CHECK-NEXT: {{[0-9a-f]+}}:   bl      g@PLT
// CHECK-NEXT: {{[0-9a-f]+}}:   add     x0, x0, #0x3
// CHECK-NEXT: {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT: {{[0-9a-f]+}}:   autib1716
// CHECK-NEXT: {{[0-9a-f]+}}:   ret
        ret
        .size f_autib1716, .-f_autib1716

        .globl  f_autiax12
        .type   f_autiax12,@function
f_autiax12:
        hint    #25
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autia   x12, sp
// CHECK-LABEL: GS-PACRET: non-protected ret found in function f_autiax12, basic block .LBB{{[0-9]+}}, at address
// CHECK-NEXT:    The return instruction is     {{[0-9a-f]+}}:       ret
// CHECK-NEXT:    The 1 instructions that write to the return register after any authentication are:
// CHECK-NEXT:    1. {{[0-9a-f]+}}: ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT: {{[0-9a-f]+}}:   paciasp
// CHECK-NEXT: {{[0-9a-f]+}}:   stp     x29, x30, [sp, #-0x10]!
// CHECK-NEXT: {{[0-9a-f]+}}:   mov     x29, sp
// CHECK-NEXT: {{[0-9a-f]+}}:   bl      g@PLT
// CHECK-NEXT: {{[0-9a-f]+}}:   add     x0, x0, #0x3
// CHECK-NEXT: {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT: {{[0-9a-f]+}}:   autia   x12, sp
// CHECK-NEXT: {{[0-9a-f]+}}:   ret
        ret
        .size f_autiax12, .-f_autiax12

        .globl  f_autibx12
        .type   f_autibx12,@function
f_autibx12:
        hint    #25
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autib   x12, sp
// CHECK-LABEL: GS-PACRET: non-protected ret found in function f_autibx12, basic block .LBB{{[0-9]+}}, at address
// CHECK-NEXT:    The return instruction is     {{[0-9a-f]+}}:       ret
// CHECK-NEXT:    The 1 instructions that write to the return register after any authentication are:
// CHECK-NEXT:    1. {{[0-9a-f]+}}: ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT: {{[0-9a-f]+}}:   paciasp
// CHECK-NEXT: {{[0-9a-f]+}}:   stp     x29, x30, [sp, #-0x10]!
// CHECK-NEXT: {{[0-9a-f]+}}:   mov     x29, sp
// CHECK-NEXT: {{[0-9a-f]+}}:   bl      g@PLT
// CHECK-NEXT: {{[0-9a-f]+}}:   add     x0, x0, #0x3
// CHECK-NEXT: {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT: {{[0-9a-f]+}}:   autib   x12, sp
// CHECK-NEXT: {{[0-9a-f]+}}:   ret
        ret
        .size f_autibx12, .-f_autibx12

        .globl  f_autiax30
        .type   f_autiax30,@function
f_autiax30:
        hint    #25
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
        hint    #25
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
        hint    #25
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autda   x12, sp
// CHECK-LABEL: GS-PACRET: non-protected ret found in function f_autdax12, basic block .LBB{{[0-9]+}}, at address
// CHECK-NEXT:    The return instruction is     {{[0-9a-f]+}}:       ret
// CHECK-NEXT:    The 1 instructions that write to the return register after any authentication are:
// CHECK-NEXT:    1. {{[0-9a-f]+}}: ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT: {{[0-9a-f]+}}:   paciasp
// CHECK-NEXT: {{[0-9a-f]+}}:   stp     x29, x30, [sp, #-0x10]!
// CHECK-NEXT: {{[0-9a-f]+}}:   mov     x29, sp
// CHECK-NEXT: {{[0-9a-f]+}}:   bl      g@PLT
// CHECK-NEXT: {{[0-9a-f]+}}:   add     x0, x0, #0x3
// CHECK-NEXT: {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT: {{[0-9a-f]+}}:   autda   x12, sp
// CHECK-NEXT: {{[0-9a-f]+}}:   ret
        ret
        .size f_autdax12, .-f_autdax12

        .globl  f_autdbx12
        .type   f_autdbx12,@function
f_autdbx12:
        hint    #25
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autdb   x12, sp
// CHECK-LABEL: GS-PACRET: non-protected ret found in function f_autdbx12, basic block .LBB{{[0-9]+}}, at address
// CHECK-NEXT:    The return instruction is     {{[0-9a-f]+}}:       ret
// CHECK-NEXT:    The 1 instructions that write to the return register after any authentication are:
// CHECK-NEXT:    1. {{[0-9a-f]+}}: ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT: {{[0-9a-f]+}}:   paciasp
// CHECK-NEXT: {{[0-9a-f]+}}:   stp     x29, x30, [sp, #-0x10]!
// CHECK-NEXT: {{[0-9a-f]+}}:   mov     x29, sp
// CHECK-NEXT: {{[0-9a-f]+}}:   bl      g@PLT
// CHECK-NEXT: {{[0-9a-f]+}}:   add     x0, x0, #0x3
// CHECK-NEXT: {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT: {{[0-9a-f]+}}:   autdb   x12, sp
// CHECK-NEXT: {{[0-9a-f]+}}:   ret
        ret
        .size f_autdbx12, .-f_autdbx12

        .globl  f_autdax30
        .type   f_autdax30,@function
f_autdax30:
        hint    #25
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
        hint    #25
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
        hint    #25
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autiza  x12
// CHECK-LABEL: GS-PACRET: non-protected ret found in function f_autizax12, basic block .LBB{{[0-9]+}}, at address
// CHECK-NEXT:    The return instruction is     {{[0-9a-f]+}}:       ret
// CHECK-NEXT:    The 1 instructions that write to the return register after any authentication are:
// CHECK-NEXT:    1. {{[0-9a-f]+}}: ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT: {{[0-9a-f]+}}:   paciasp
// CHECK-NEXT: {{[0-9a-f]+}}:   stp     x29, x30, [sp, #-0x10]!
// CHECK-NEXT: {{[0-9a-f]+}}:   mov     x29, sp
// CHECK-NEXT: {{[0-9a-f]+}}:   bl      g@PLT
// CHECK-NEXT: {{[0-9a-f]+}}:   add     x0, x0, #0x3
// CHECK-NEXT: {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT: {{[0-9a-f]+}}:   autiza  x12
// CHECK-NEXT: {{[0-9a-f]+}}:   ret
        ret
        .size f_autizax12, .-f_autizax12

        .globl  f_autizbx12
        .type   f_autizbx12,@function
f_autizbx12:
        hint    #25
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autizb  x12
// CHECK-LABEL: GS-PACRET: non-protected ret found in function f_autizbx12, basic block .LBB{{[0-9]+}}, at address
// CHECK-NEXT:    The return instruction is     {{[0-9a-f]+}}:       ret
// CHECK-NEXT:    The 1 instructions that write to the return register after any authentication are:
// CHECK-NEXT:    1. {{[0-9a-f]+}}: ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT: {{[0-9a-f]+}}:   paciasp
// CHECK-NEXT: {{[0-9a-f]+}}:   stp     x29, x30, [sp, #-0x10]!
// CHECK-NEXT: {{[0-9a-f]+}}:   mov     x29, sp
// CHECK-NEXT: {{[0-9a-f]+}}:   bl      g@PLT
// CHECK-NEXT: {{[0-9a-f]+}}:   add     x0, x0, #0x3
// CHECK-NEXT: {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT: {{[0-9a-f]+}}:   autizb  x12
// CHECK-NEXT: {{[0-9a-f]+}}:   ret
        ret
        .size f_autizbx12, .-f_autizbx12

        .globl  f_autizax30
        .type   f_autizax30,@function
f_autizax30:
        hint    #25
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
        hint    #25
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
        hint    #25
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autdza  x12
// CHECK-LABEL: GS-PACRET: non-protected ret found in function f_autdzax12, basic block .LBB{{[0-9]+}}, at address
// CHECK-NEXT:    The return instruction is     {{[0-9a-f]+}}:       ret
// CHECK-NEXT:    The 1 instructions that write to the return register after any authentication are:
// CHECK-NEXT:    1. {{[0-9a-f]+}}: ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT: {{[0-9a-f]+}}:   paciasp
// CHECK-NEXT: {{[0-9a-f]+}}:   stp     x29, x30, [sp, #-0x10]!
// CHECK-NEXT: {{[0-9a-f]+}}:   mov     x29, sp
// CHECK-NEXT: {{[0-9a-f]+}}:   bl      g@PLT
// CHECK-NEXT: {{[0-9a-f]+}}:   add     x0, x0, #0x3
// CHECK-NEXT: {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT: {{[0-9a-f]+}}:   autdza  x12
// CHECK-NEXT: {{[0-9a-f]+}}:   ret
        ret
        .size f_autdzax12, .-f_autdzax12

        .globl  f_autdzbx12
        .type   f_autdzbx12,@function
f_autdzbx12:
        hint    #25
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
        autdzb  x12
// CHECK-LABEL: GS-PACRET: non-protected ret found in function f_autdzbx12, basic block .LBB{{[0-9]+}}, at address
// CHECK-NEXT:    The return instruction is     {{[0-9a-f]+}}:       ret
// CHECK-NEXT:    The 1 instructions that write to the return register after any authentication are:
// CHECK-NEXT:    1. {{[0-9a-f]+}}: ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT: {{[0-9a-f]+}}:   paciasp
// CHECK-NEXT: {{[0-9a-f]+}}:   stp     x29, x30, [sp, #-0x10]!
// CHECK-NEXT: {{[0-9a-f]+}}:   mov     x29, sp
// CHECK-NEXT: {{[0-9a-f]+}}:   bl      g@PLT
// CHECK-NEXT: {{[0-9a-f]+}}:   add     x0, x0, #0x3
// CHECK-NEXT: {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT: {{[0-9a-f]+}}:   autdzb  x12
// CHECK-NEXT: {{[0-9a-f]+}}:   ret
        ret
        .size f_autdzbx12, .-f_autdzbx12

        .globl  f_autdzax30
        .type   f_autdzax30,@function
f_autdzax30:
        hint    #25
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
        hint    #25
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
        hint    #25
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
        hint    #25
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
        hint    #25
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
// CHECK-LABEL: GS-PACRET: Warning: pac-ret analysis could not analyze this return instruction in function f_eretaa, basic block .LBB{{[0-9]+}}, at address
// CHECK-NEXT:   The return instruction is     {{[0-9a-f]+}}:       eretaa
        eretaa
        .size f_eretaa, .-f_eretaa

        .globl  f_eretab
        .type   f_eretab,@function
f_eretab:
        hint    #25
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
// CHECK-LABEL: GS-PACRET: Warning: pac-ret analysis could not analyze this return instruction in function f_eretab, basic block .LBB{{[0-9]+}}, at address
// CHECK-NEXT:   The return instruction is     {{[0-9a-f]+}}:       eretab
        eretab
        .size f_eretab, .-f_eretab

        .globl  f_eret
        .type   f_eret,@function
f_eret:
        hint    #25
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        bl      g
        add     x0, x0, #3
        ldp     x29, x30, [sp], #16
// CHECK-LABEL: GS-PACRET: Warning: pac-ret analysis could not analyze this return instruction in function f_eret, basic block .LBB{{[0-9]+}}, at address
// CHECK-NEXT:   The return instruction is     {{[0-9a-f]+}}:       eret
        eret
        .size f_eret, .-f_eret

        .globl f_movx30reg
        .type   f_movx30reg,@function
f_movx30reg:
// CHECK-LABEL: GS-PACRET: non-protected ret found in function f_movx30reg, basic block .LBB{{[0-9]+}}, at address
// CHECK-NEXT:    The return instruction is     {{[0-9a-f]+}}:       ret
// CHECK-NEXT:    The 1 instructions that write to the return register after any authentication are:
// CHECK-NEXT:    1. {{[0-9a-f]+}}: mov x30, x22
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT: {{[0-9a-f]+}}:   mov     x30, x22
// CHECK-NEXT: {{[0-9a-f]+}}:   ret
        mov     x30, x22
        ret
        .size f_movx30reg, .-f_movx30reg

// FIXME: add regression tests for the instructions added in v9.5: AUTI{A,B}SPPC{i,r}, RETI{A,B}SPPC{i,r}, AUTI{A,B}171615.


// TODO: add test to see if registers clobbered by a call are picked up.
