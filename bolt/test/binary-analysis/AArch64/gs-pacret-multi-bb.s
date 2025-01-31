// RUN: %clang %cflags -march=armv8.3-a -mbranch-protection=pac-ret %s %p/../../Inputs/asm_main.c -o %t.exe
// RUN: llvm-bolt-binary-analysis --scanners=pacret %t.exe 2>&1 | FileCheck %s


// Verify that we can also detect gadgets across basic blocks

        .globl f_crossbb1
        .type   f_crossbb1,@function
f_crossbb1:
        paciasp
        stp     x29, x30, [sp, #-16]!
        ldp     x29, x30, [sp], #16
        cbnz    x0, 1f
        autiasp
1:
        ret
        .size f_crossbb1, .-f_crossbb1
// CHECK-LABEL:     GS-PACRET: non-protected ret found in function f_crossbb1, basic block .L{{[^,]+}}, at address
// CHECK-NEXT:  The return instruction is     {{[0-9a-f]+}}:       ret
// CHECK-NEXT:  The 2 instructions that write to the return register after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  2.     {{[0-9a-f]+}}:      autiasp

// A test that checks that the dataflow state tracking across when merging BBs
// seems to work:
        .globl f_mergebb1
        .type   f_mergebb1,@function
f_mergebb1:
        paciasp
2:
        stp     x29, x30, [sp, #-16]!
        ldp     x29, x30, [sp], #16
        sub     x0, x0, #1
        cbnz    x0, 1f
        autiasp
        b       2b
1:
        ret
        .size f_mergebb1, .-f_mergebb1
// CHECK-LABEL: GS-PACRET: non-protected ret found in function f_mergebb1, basic block .L{{[^,]+}}, at address
// CHECK-NEXT:    The return instruction is     {{[0-9a-f]+}}:       ret
// CHECK-NEXT:    The 1 instructions that write to the return register after any authentication are:
// CHECK-NEXT:    1.     {{[0-9a-f]+}}:      ldp     x29, x30, [sp], #0x10

        .globl f_shrinkwrapping
        .type   f_shrinkwrapping,@function
f_shrinkwrapping:
        cbz     x0, 1f
        paciasp
        stp     x29, x30, [sp, #-16]!
        ldp     x29, x30, [sp], #16
        autiasp
1:
        ret
        .size f_shrinkwrapping, .-f_shrinkwrapping
// CHECK-NOT: f_shrinkwrapping

        .globl f_multi_auth_insts
        .type   f_multi_auth_insts,@function
f_multi_auth_insts:
        paciasp
        stp     x29, x30, [sp, #-16]!
        ldp     x29, x30, [sp], #16
        cbnz x0, 1f
        autibsp
        b 2f
1:
        autiasp
2:
        ret
        .size f_multi_auth_insts, .-f_multi_auth_insts
// CHECK-NOT: f_multi_auth_insts

// TODO: also verify that false negatives exist in across-BB gadgets in functions
// for which bolt cannot reconstruct the call graph.
