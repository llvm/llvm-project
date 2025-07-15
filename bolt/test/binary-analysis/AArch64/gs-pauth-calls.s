// RUN: %clang %cflags -march=armv8.3-a %s -o %t.exe
// RUN: llvm-bolt-binary-analysis --scanners=pacret %t.exe 2>&1 | FileCheck -check-prefix=PACRET %s
// RUN: llvm-bolt-binary-analysis --scanners=pauth %t.exe 2>&1 | FileCheck %s

// PACRET-NOT: non-protected call found in function

        .text

        .globl  callee
        .type   callee,@function
callee:
        ret
        .size callee, .-callee

        .globl  good_direct_call
        .type   good_direct_call,@function
good_direct_call:
// CHECK-NOT: good_direct_call
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        bl      callee

        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size good_direct_call, .-good_direct_call

        .globl  good_indirect_call_arg
        .type   good_indirect_call_arg,@function
good_indirect_call_arg:
// CHECK-NOT: good_indirect_call_arg
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        autia   x0, x1
        blr     x0

        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size good_indirect_call_arg, .-good_indirect_call_arg

        .globl  good_indirect_call_mem
        .type   good_indirect_call_mem,@function
good_indirect_call_mem:
// CHECK-NOT: good_indirect_call_mem
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        ldr     x16, [x0]
        autia   x16, x0
        blr     x16

        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size good_indirect_call_mem, .-good_indirect_call_mem

        .globl  good_indirect_call_arg_v83
        .type   good_indirect_call_arg_v83,@function
good_indirect_call_arg_v83:
// CHECK-NOT: good_indirect_call_arg_v83
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        blraa   x0, x1

        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size good_indirect_call_arg_v83, .-good_indirect_call_arg_v83

        .globl  good_indirect_call_mem_v83
        .type   good_indirect_call_mem_v83,@function
good_indirect_call_mem_v83:
// CHECK-NOT: good_indirect_call_mem_v83
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        ldr     x16, [x0]
        blraa   x16, x0

        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size good_indirect_call_mem_v83, .-good_indirect_call_mem_v83

        .globl  bad_indirect_call_arg
        .type   bad_indirect_call_arg,@function
bad_indirect_call_arg:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function bad_indirect_call_arg, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:         blr     x0
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        blr     x0

        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size bad_indirect_call_arg, .-bad_indirect_call_arg

        .globl  bad_indirect_call_mem
        .type   bad_indirect_call_mem,@function
bad_indirect_call_mem:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function bad_indirect_call_mem, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:         blr     x16
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      ldr     x16, [x0]
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   paciasp
// CHECK-NEXT:  {{[0-9a-f]+}}:   stp     x29, x30, [sp, #-0x10]!
// CHECK-NEXT:  {{[0-9a-f]+}}:   mov     x29, sp
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldr     x16, [x0]
// CHECK-NEXT:  {{[0-9a-f]+}}:   blr     x16
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  {{[0-9a-f]+}}:   autiasp
// CHECK-NEXT:  {{[0-9a-f]+}}:   ret
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        ldr     x16, [x0]
        blr     x16

        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size bad_indirect_call_mem, .-bad_indirect_call_mem

        .globl  bad_indirect_call_arg_clobber
        .type   bad_indirect_call_arg_clobber,@function
bad_indirect_call_arg_clobber:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function bad_indirect_call_arg_clobber, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:         blr     x0
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      mov     w0, w2
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   paciasp
// CHECK-NEXT:  {{[0-9a-f]+}}:   stp     x29, x30, [sp, #-0x10]!
// CHECK-NEXT:  {{[0-9a-f]+}}:   mov     x29, sp
// CHECK-NEXT:  {{[0-9a-f]+}}:   autia   x0, x1
// CHECK-NEXT:  {{[0-9a-f]+}}:   mov     w0, w2
// CHECK-NEXT:  {{[0-9a-f]+}}:   blr     x0
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  {{[0-9a-f]+}}:   autiasp
// CHECK-NEXT:  {{[0-9a-f]+}}:   ret
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        autia   x0, x1
        mov     w0, w2
        blr     x0

        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size bad_indirect_call_arg_clobber, .-bad_indirect_call_arg_clobber

        .globl  bad_indirect_call_mem_clobber
        .type   bad_indirect_call_mem_clobber,@function
bad_indirect_call_mem_clobber:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function bad_indirect_call_mem_clobber, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:         blr     x16
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      mov     w16, w2
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   paciasp
// CHECK-NEXT:  {{[0-9a-f]+}}:   stp     x29, x30, [sp, #-0x10]!
// CHECK-NEXT:  {{[0-9a-f]+}}:   mov     x29, sp
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldr     x16, [x0]
// CHECK-NEXT:  {{[0-9a-f]+}}:   autia   x16, x0
// CHECK-NEXT:  {{[0-9a-f]+}}:   mov     w16, w2
// CHECK-NEXT:  {{[0-9a-f]+}}:   blr     x16
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  {{[0-9a-f]+}}:   autiasp
// CHECK-NEXT:  {{[0-9a-f]+}}:   ret
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        ldr     x16, [x0]
        autia   x16, x0
        mov     w16, w2
        blr     x16

        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size bad_indirect_call_mem_clobber, .-bad_indirect_call_mem_clobber

        .globl  good_indirect_call_mem_chain_of_auts
        .type   good_indirect_call_mem_chain_of_auts,@function
good_indirect_call_mem_chain_of_auts:
// CHECK-NOT: good_indirect_call_mem_chain_of_auts
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        ldr     x16, [x0]
        autda   x16, x1
        ldr     x16, [x16]
        autia   x16, x0
        blr     x16

        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size good_indirect_call_mem_chain_of_auts, .-good_indirect_call_mem_chain_of_auts

        .globl  bad_indirect_call_mem_chain_of_auts
        .type   bad_indirect_call_mem_chain_of_auts,@function
bad_indirect_call_mem_chain_of_auts:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function bad_indirect_call_mem_chain_of_auts, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:         blr     x16
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      ldr     x16, [x16]
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   paciasp
// CHECK-NEXT:  {{[0-9a-f]+}}:   stp     x29, x30, [sp, #-0x10]!
// CHECK-NEXT:  {{[0-9a-f]+}}:   mov     x29, sp
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldr     x16, [x0]
// CHECK-NEXT:  {{[0-9a-f]+}}:   autda   x16, x1
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldr     x16, [x16]
// CHECK-NEXT:  {{[0-9a-f]+}}:   blr     x16
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  {{[0-9a-f]+}}:   autiasp
// CHECK-NEXT:  {{[0-9a-f]+}}:   ret
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        ldr     x16, [x0]
        autda   x16, x1
        ldr     x16, [x16]
        // Missing AUT of x16. The fact that x16 was authenticated above has nothing to do with it.
        blr     x16

        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size bad_indirect_call_mem_chain_of_auts, .-bad_indirect_call_mem_chain_of_auts

// Multi-BB test cases.
//
// Positive ("good") test cases are designed so that the register is made safe
// in one BB and used in the other. Negative ("bad") ones are designed so that
// there are two predecessors, one of them ends with the register in a safe
// state and the other ends with that register being unsafe.

        .globl  good_indirect_call_arg_multi_bb
        .type   good_indirect_call_arg_multi_bb,@function
good_indirect_call_arg_multi_bb:
// CHECK-NOT: good_indirect_call_arg_multi_bb
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        autia   x0, x1
        cbz     x2, 1f
        blr     x0
1:
        ldr     x1, [x0]  // prevent authentication oracle

        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size good_indirect_call_arg_multi_bb, .-good_indirect_call_arg_multi_bb

        .globl  good_indirect_call_mem_multi_bb
        .type   good_indirect_call_mem_multi_bb,@function
good_indirect_call_mem_multi_bb:
// CHECK-NOT: good_indirect_call_mem_multi_bb
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        ldr     x16, [x0]
        autia   x16, x0
        cbz     x2, 1f
        blr     x16
1:
        ldr     w0, [x16]  // prevent authentication oracle

        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size good_indirect_call_mem_multi_bb, .-good_indirect_call_mem_multi_bb

        .globl  bad_indirect_call_arg_multi_bb
        .type   bad_indirect_call_arg_multi_bb,@function
bad_indirect_call_arg_multi_bb:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function bad_indirect_call_arg_multi_bb, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:         blr     x0
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        cbz     x2, 1f
        autia   x0, x1
1:
        blr     x0

        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size bad_indirect_call_arg_multi_bb, .-bad_indirect_call_arg_multi_bb

        .globl  bad_indirect_call_mem_multi_bb
        .type   bad_indirect_call_mem_multi_bb,@function
bad_indirect_call_mem_multi_bb:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function bad_indirect_call_mem_multi_bb, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:         blr     x16
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      ldr     x16, [x0]
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        ldr     x16, [x0]
        cbz     x2, 1f
        autia   x16, x1
1:
        blr     x16

        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size bad_indirect_call_mem_multi_bb, .-bad_indirect_call_mem_multi_bb

        .globl  bad_indirect_call_arg_clobber_multi_bb
        .type   bad_indirect_call_arg_clobber_multi_bb,@function
bad_indirect_call_arg_clobber_multi_bb:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function bad_indirect_call_arg_clobber_multi_bb, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:         blr     x0
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      mov     w0, w3
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        autia   x0, x1
        cbz     x2, 1f
        mov     w0, w3
1:
        blr     x0

        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size bad_indirect_call_arg_clobber_multi_bb, .-bad_indirect_call_arg_clobber_multi_bb

        .globl  bad_indirect_call_mem_clobber_multi_bb
        .type   bad_indirect_call_mem_clobber_multi_bb,@function
bad_indirect_call_mem_clobber_multi_bb:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function bad_indirect_call_mem_clobber_multi_bb, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:         blr     x16
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      mov     w16, w2
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        ldr     x16, [x0]
        autia   x16, x0
        cbz     x2, 1f
        mov     w16, w2
1:
        blr     x16

        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size bad_indirect_call_mem_clobber_multi_bb, .-bad_indirect_call_mem_clobber_multi_bb

        .globl  good_indirect_call_mem_chain_of_auts_multi_bb
        .type   good_indirect_call_mem_chain_of_auts_multi_bb,@function
good_indirect_call_mem_chain_of_auts_multi_bb:
// CHECK-NOT: good_indirect_call_mem_chain_of_auts_multi_bb
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        ldr     x16, [x0]
        autda   x16, x1
        ldr     x16, [x16]
        autia   x16, x0
        cbz     x2, 1f
        blr     x16
1:
        ldr     w0, [x16]  // prevent authentication oracle

        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size good_indirect_call_mem_chain_of_auts_multi_bb, .-good_indirect_call_mem_chain_of_auts_multi_bb

        .globl  bad_indirect_call_mem_chain_of_auts_multi_bb
        .type   bad_indirect_call_mem_chain_of_auts_multi_bb,@function
bad_indirect_call_mem_chain_of_auts_multi_bb:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function bad_indirect_call_mem_chain_of_auts_multi_bb, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:         blr     x16
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      ldr     x16, [x16]
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        ldr     x16, [x0]
        autda   x16, x1
        ldr     x16, [x16]
        cbz     x2, 1f
        autia   x16, x0
1:
        blr     x16

        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size bad_indirect_call_mem_chain_of_auts_multi_bb, .-bad_indirect_call_mem_chain_of_auts_multi_bb

// Tests for CFG-unaware analysis.
//
// All these tests use an instruction sequence like this
//
//      adr x2, 1f
//      br  x2
//    1:
//      ; ...
//
// to make BOLT unable to reconstruct the control flow. Note that one can easily
// tell whether the report corresponds to a function with or without CFG:
// normally, the location of the gadget is described like this:
//
//     ... found in function <function_name>, basic block <basic block name>, at address <address>
//
// When CFG information is not available, this is reduced to
//
//     ... found in function <function_name>, at address <address>

        .globl  good_direct_call_nocfg
        .type   good_direct_call_nocfg,@function
good_direct_call_nocfg:
// CHECK-NOT: good_direct_call_nocfg
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        bl      callee

        adr     x2, 1f
        br      x2
1:
        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size good_direct_call_nocfg, .-good_direct_call_nocfg

        .globl  good_indirect_call_arg_nocfg
        .type   good_indirect_call_arg_nocfg,@function
good_indirect_call_arg_nocfg:
// CHECK-NOT: good_indirect_call_arg_nocfg
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        autia   x0, x1
        blr     x0

        adr     x2, 1f
        br      x2
1:
        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size good_indirect_call_arg_nocfg, .-good_indirect_call_arg_nocfg

        .globl  good_indirect_call_mem_nocfg
        .type   good_indirect_call_mem_nocfg,@function
good_indirect_call_mem_nocfg:
// CHECK-NOT: good_indirect_call_mem_nocfg
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        ldr     x16, [x0]
        autia   x16, x0
        blr     x16

        adr     x2, 1f
        br      x2
1:
        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size good_indirect_call_mem_nocfg, .-good_indirect_call_mem_nocfg

        .globl  good_indirect_call_arg_v83_nocfg
        .type   good_indirect_call_arg_v83_nocfg,@function
good_indirect_call_arg_v83_nocfg:
// CHECK-NOT: good_indirect_call_arg_v83_nocfg
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        blraa   x0, x1

        adr     x2, 1f
        br      x2
1:
        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size good_indirect_call_arg_v83_nocfg, .-good_indirect_call_arg_v83_nocfg

        .globl  good_indirect_call_mem_v83_nocfg
        .type   good_indirect_call_mem_v83_nocfg,@function
good_indirect_call_mem_v83_nocfg:
// CHECK-NOT: good_indirect_call_mem_v83_nocfg
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        ldr     x16, [x0]
        blraa   x16, x0

        adr     x2, 1f
        br      x2
1:
        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size good_indirect_call_mem_v83_nocfg, .-good_indirect_call_mem_v83_nocfg

        .globl  bad_indirect_call_arg_nocfg
        .type   bad_indirect_call_arg_nocfg,@function
bad_indirect_call_arg_nocfg:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function bad_indirect_call_arg_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:         blr     x0
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        blr     x0

        adr     x2, 1f
        br      x2
1:
        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size bad_indirect_call_arg_nocfg, .-bad_indirect_call_arg_nocfg

        .globl  obscure_indirect_call_arg_nocfg
        .type   obscure_indirect_call_arg_nocfg,@function
obscure_indirect_call_arg_nocfg:
// CHECK-NOCFG-LABEL: GS-PAUTH: non-protected call found in function obscure_indirect_call_arg_nocfg, at address
// CHECK-NOCFG-NEXT:  The instruction is     {{[0-9a-f]+}}:         blr     x0
// CHECK-NOCFG-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        autia   x0, x1 // not observed by the checker
        b       1f
1:
        // The register state is pessimistically reset after a label, thus
        // the below branch instruction is reported as non-protected - this is
        // a known false-positive.
        blr     x0

        adr     x2, 1f
        br      x2
1:
        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size obscure_good_indirect_call_arg_nocfg, .-obscure_good_indirect_call_arg_nocfg

        .globl  safe_lr_at_function_entry_nocfg
        .type   safe_lr_at_function_entry_nocfg,@function
safe_lr_at_function_entry_nocfg:
// Due to state being reset after a label, paciasp is reported as
// a signing oracle - this is a known false positive, ignore it.
// CHECK-NOT: non-protected call{{.*}}safe_lr_at_function_entry_nocfg
        cbz     x0, 1f
        ret                            // LR is safe at the start of the function
1:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        adr     x2, 2f
        br      x2
2:
        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size safe_lr_at_function_entry_nocfg, .-safe_lr_at_function_entry_nocfg

        .globl  lr_is_never_unsafe_before_first_inst_nocfg
        .type   lr_is_never_unsafe_before_first_inst_nocfg,@function
// CHECK-NOT: lr_is_never_unsafe_before_first_inst_nocfg
lr_is_never_unsafe_before_first_inst_nocfg:
1:
        // The register state is never reset before the first instruction of
        // the function. This can lead to a known false-negative if LR is
        // clobbered and then a jump to the very first instruction of the
        // function is performed.
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        mov     x30, x0
        cbz     x1, 1b

        adr     x2, 2f
        br      x2
2:
        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size lr_is_never_unsafe_before_first_inst_nocfg, .-lr_is_never_unsafe_before_first_inst_nocfg

        .globl  bad_indirect_call_mem_nocfg
        .type   bad_indirect_call_mem_nocfg,@function
bad_indirect_call_mem_nocfg:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function bad_indirect_call_mem_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:         blr     x16
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      ldr     x16, [x0]
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        ldr     x16, [x0]
        blr     x16

        adr     x2, 1f
        br      x2
1:
        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size bad_indirect_call_mem_nocfg, .-bad_indirect_call_mem_nocfg

        .globl  bad_indirect_call_arg_clobber_nocfg
        .type   bad_indirect_call_arg_clobber_nocfg,@function
bad_indirect_call_arg_clobber_nocfg:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function bad_indirect_call_arg_clobber_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:         blr     x0
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      mov     w0, w2
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        autia   x0, x1
        mov     w0, w2
        blr     x0

        adr     x2, 1f
        br      x2
1:
        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size bad_indirect_call_arg_clobber_nocfg, .-bad_indirect_call_arg_clobber_nocfg

        .globl  bad_indirect_call_mem_clobber_nocfg
        .type   bad_indirect_call_mem_clobber_nocfg,@function
bad_indirect_call_mem_clobber_nocfg:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function bad_indirect_call_mem_clobber_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:         blr     x16
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      mov     w16, w2
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        ldr     x16, [x0]
        autia   x16, x0
        mov     w16, w2
        blr     x16

        adr     x2, 1f
        br      x2
1:
        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size bad_indirect_call_mem_clobber_nocfg, .-bad_indirect_call_mem_clobber_nocfg

        .globl  good_indirect_call_mem_chain_of_auts_nocfg
        .type   good_indirect_call_mem_chain_of_auts_nocfg,@function
good_indirect_call_mem_chain_of_auts_nocfg:
// CHECK-NOT: good_indirect_call_mem_chain_of_auts_nocfg
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        ldr     x16, [x0]
        autda   x16, x1
        ldr     x16, [x16]
        autia   x16, x0
        blr     x16

        adr     x2, 1f
        br      x2
1:
        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size good_indirect_call_mem_chain_of_auts_nocfg, .-good_indirect_call_mem_chain_of_auts_nocfg

        .globl  bad_indirect_call_mem_chain_of_auts_nocfg
        .type   bad_indirect_call_mem_chain_of_auts_nocfg,@function
bad_indirect_call_mem_chain_of_auts_nocfg:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function bad_indirect_call_mem_chain_of_auts_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:         blr     x16
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      ldr     x16, [x16]
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        ldr     x16, [x0]
        autda   x16, x1
        ldr     x16, [x16]
        // Missing AUT of x16. The fact that x16 was authenticated above has nothing to do with it.
        blr     x16

        adr     x2, 1f
        br      x2
1:
        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size bad_indirect_call_mem_chain_of_auts_nocfg, .-bad_indirect_call_mem_chain_of_auts_nocfg

// Test tail calls. To somewhat decrease the number of test cases and not
// duplicate all of the above, only implement "mem" variant of test cases and
// mostly test negative cases.

        .globl  good_direct_tailcall
        .type   good_direct_tailcall,@function
good_direct_tailcall:
// CHECK-NOT: good_direct_tailcall
        b       callee
        .size good_direct_tailcall, .-good_direct_tailcall

        .globl  good_indirect_tailcall_mem
        .type   good_indirect_tailcall_mem,@function
good_indirect_tailcall_mem:
// CHECK-NOT: good_indirect_tailcall_mem
        ldr     x16, [x0]
        autia   x16, x0
        br      x16
        .size good_indirect_tailcall_mem, .-good_indirect_tailcall_mem

        .globl  good_indirect_tailcall_mem_v83
        .type   good_indirect_tailcall_mem_v83,@function
good_indirect_tailcall_mem_v83:
// CHECK-NOT: good_indirect_tailcall_mem_v83
        ldr     x16, [x0]
        braa    x16, x0
        .size good_indirect_tailcall_mem_v83, .-good_indirect_tailcall_mem_v83

        .globl  bad_indirect_tailcall_mem
        .type   bad_indirect_tailcall_mem,@function
bad_indirect_tailcall_mem:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function bad_indirect_tailcall_mem, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:         br      x16
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      ldr     x16, [x0]
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldr     x16, [x0]
// CHECK-NEXT:  {{[0-9a-f]+}}:   br      x16
        ldr     x16, [x0]
        br      x16
        .size bad_indirect_tailcall_mem, .-bad_indirect_tailcall_mem

        .globl  bad_indirect_tailcall_mem_clobber
        .type   bad_indirect_tailcall_mem_clobber,@function
bad_indirect_tailcall_mem_clobber:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function bad_indirect_tailcall_mem_clobber, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:         br      x16
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      mov     w16, w2
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldr     x16, [x0]
// CHECK-NEXT:  {{[0-9a-f]+}}:   autia   x16, x0
// CHECK-NEXT:  {{[0-9a-f]+}}:   mov     w16, w2
// CHECK-NEXT:  {{[0-9a-f]+}}:   br      x16
        ldr     x16, [x0]
        autia   x16, x0
        mov     w16, w2
        br      x16
        .size bad_indirect_tailcall_mem_clobber, .-bad_indirect_tailcall_mem_clobber

        .globl  bad_indirect_tailcall_mem_chain_of_auts
        .type   bad_indirect_tailcall_mem_chain_of_auts,@function
bad_indirect_tailcall_mem_chain_of_auts:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function bad_indirect_tailcall_mem_chain_of_auts, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:         br      x16
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      ldr     x16, [x16]
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldr     x16, [x0]
// CHECK-NEXT:  {{[0-9a-f]+}}:   autda   x16, x1
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldr     x16, [x16]
// CHECK-NEXT:  {{[0-9a-f]+}}:   br      x16
        ldr     x16, [x0]
        autda   x16, x1
        ldr     x16, [x16]
        // Missing AUT of x16. The fact that x16 was authenticated above has nothing to do with it.
        br      x16
        .size bad_indirect_tailcall_mem_chain_of_auts, .-bad_indirect_tailcall_mem_chain_of_auts

        .globl  bad_indirect_tailcall_mem_multi_bb
        .type   bad_indirect_tailcall_mem_multi_bb,@function
bad_indirect_tailcall_mem_multi_bb:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function bad_indirect_tailcall_mem_multi_bb, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:         br      x16
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      ldr     x16, [x0]
        ldr     x16, [x0]
        cbz     x2, 1f
        autia   x16, x1
1:
        br      x16
        .size bad_indirect_tailcall_mem_multi_bb, .-bad_indirect_tailcall_mem_multi_bb

        .globl  bad_indirect_tailcall_mem_clobber_multi_bb
        .type   bad_indirect_tailcall_mem_clobber_multi_bb,@function
bad_indirect_tailcall_mem_clobber_multi_bb:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function bad_indirect_tailcall_mem_clobber_multi_bb, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:         br      x16
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      mov     w16, w2
        ldr     x16, [x0]
        autia   x16, x0
        cbz     x2, 1f
        mov     w16, w2
1:
        br      x16
        .size bad_indirect_tailcall_mem_clobber_multi_bb, .-bad_indirect_tailcall_mem_clobber_multi_bb

        .globl  good_direct_tailcall_nocfg
        .type   good_direct_tailcall_nocfg,@function
good_direct_tailcall_nocfg:
// CHECK-NOT: good_direct_tailcall_nocfg
        adr     x2, 1f
        br      x2
1:
        b       callee
        .size good_direct_tailcall_nocfg, .-good_direct_tailcall_nocfg

        .globl  good_indirect_tailcall_mem_nocfg
        .type   good_indirect_tailcall_mem_nocfg,@function
good_indirect_tailcall_mem_nocfg:
// CHECK-NOT: good_indirect_tailcall_mem_nocfg
        adr     x2, 1f
        br      x2
1:
        ldr     x16, [x0]
        autia   x16, x0
        br      x16
        .size good_indirect_tailcall_mem_nocfg, .-good_indirect_tailcall_mem_nocfg

        .globl  good_indirect_tailcall_mem_v83_nocfg
        .type   good_indirect_tailcall_mem_v83_nocfg,@function
good_indirect_tailcall_mem_v83_nocfg:
// CHECK-NOT: good_indirect_tailcall_mem_v83_nocfg
        adr     x2, 1f
        br      x2
1:
        ldr     x16, [x0]
        braa    x16, x0
        .size good_indirect_tailcall_mem_v83_nocfg, .-good_indirect_tailcall_mem_v83_nocfg

        .globl  bad_indirect_tailcall_mem_nocfg
        .type   bad_indirect_tailcall_mem_nocfg,@function
bad_indirect_tailcall_mem_nocfg:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function bad_indirect_tailcall_mem_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:         br      x16
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      ldr     x16, [x0]
        adr     x2, 1f
        br      x2
1:
        ldr     x16, [x0]
        br      x16
        .size bad_indirect_tailcall_mem_nocfg, .-bad_indirect_tailcall_mem_nocfg

        .globl  bad_indirect_tailcall_mem_clobber_nocfg
        .type   bad_indirect_tailcall_mem_clobber_nocfg,@function
bad_indirect_tailcall_mem_clobber_nocfg:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function bad_indirect_tailcall_mem_clobber_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:         br      x16
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      mov     w16, w2
        adr     x2, 1f
        br      x2
1:
        ldr     x16, [x0]
        autia   x16, x0
        mov     w16, w2
        br      x16
        .size bad_indirect_tailcall_mem_clobber_nocfg, .-bad_indirect_tailcall_mem_clobber_nocfg

        .globl  bad_indirect_tailcall_mem_chain_of_auts_nocfg
        .type   bad_indirect_tailcall_mem_chain_of_auts_nocfg,@function
bad_indirect_tailcall_mem_chain_of_auts_nocfg:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function bad_indirect_tailcall_mem_chain_of_auts_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:         br      x16
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      ldr     x16, [x16]
        adr     x2, 1f
        br      x2
1:
        ldr     x16, [x0]
        autda   x16, x1
        ldr     x16, [x16]
        // Missing AUT of x16. The fact that x16 was authenticated above has nothing to do with it.
        br      x16
        .size bad_indirect_tailcall_mem_chain_of_auts_nocfg, .-bad_indirect_tailcall_mem_chain_of_auts_nocfg

        .globl  state_is_reset_at_branch_destination_nocfg
        .type   state_is_reset_at_branch_destination_nocfg,@function
state_is_reset_at_branch_destination_nocfg:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function state_is_reset_at_branch_destination_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:         blr      x0
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        b       1f
        autia   x0, x1  // skipped
1:
        blr     x0

        adr     x2, 2f
        br      x2
2:
        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size state_is_reset_at_branch_destination_nocfg, .-state_is_reset_at_branch_destination_nocfg

// Test that calling a function is considered as invalidating safety of every
// register. Note that we only have to consider "returning" function calls
// (via branch-with-link), but both direct and indirect variants.
// Checking different registers:
// * x2 - function argument
// * x8 - indirect result location
// * x10 - temporary
// * x16 - intra-procedure-call scratch register
// * x18 - platform register
// * x20 - callee-saved register

        .globl  direct_call_invalidates_safety
        .type   direct_call_invalidates_safety,@function
direct_call_invalidates_safety:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function direct_call_invalidates_safety, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      blr     x2
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      bl      callee
// CHECK-LABEL: GS-PAUTH: non-protected call found in function direct_call_invalidates_safety, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      blr     x8
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      bl      callee
// CHECK-LABEL: GS-PAUTH: non-protected call found in function direct_call_invalidates_safety, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      blr     x10
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      bl      callee
// CHECK-LABEL: GS-PAUTH: non-protected call found in function direct_call_invalidates_safety, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      blr     x16
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      bl      callee
// CHECK-LABEL: GS-PAUTH: non-protected call found in function direct_call_invalidates_safety, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      blr     x18
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      bl      callee
// CHECK-LABEL: GS-PAUTH: non-protected call found in function direct_call_invalidates_safety, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      blr     x20
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      bl      callee
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        mov     x2, x0
        autiza  x2
        bl      callee
        blr     x2

        mov     x8, x0
        autiza  x8
        bl      callee
        blr     x8

        mov     x10, x0
        autiza  x10
        bl      callee
        blr     x10

        mov     x16, x0
        autiza  x16
        bl      callee
        blr     x16

        mov     x18, x0
        autiza  x18
        bl      callee
        blr     x18

        mov     x20, x0
        autiza  x20
        bl      callee
        blr     x20

        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size direct_call_invalidates_safety, .-direct_call_invalidates_safety

        .globl  indirect_call_invalidates_safety
        .type   indirect_call_invalidates_safety,@function
indirect_call_invalidates_safety:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function indirect_call_invalidates_safety, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      blr     x2
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      blr     x2
// Check that only one error is reported per pair of BLRs.
// CHECK-NOT:   The instruction is     {{[0-9a-f]+}}:      blr     x2

// CHECK-LABEL: GS-PAUTH: non-protected call found in function indirect_call_invalidates_safety, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      blr     x8
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      blr     x8
// CHECK-NOT:   The instruction is     {{[0-9a-f]+}}:      blr     x8

// CHECK-LABEL: GS-PAUTH: non-protected call found in function indirect_call_invalidates_safety, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      blr     x10
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      blr     x10
// CHECK-NOT:   The instruction is     {{[0-9a-f]+}}:      blr     x10

// CHECK-LABEL: GS-PAUTH: non-protected call found in function indirect_call_invalidates_safety, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      blr     x16
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      blr     x16
// CHECK-NOT:   The instruction is     {{[0-9a-f]+}}:      blr     x16

// CHECK-LABEL: GS-PAUTH: non-protected call found in function indirect_call_invalidates_safety, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      blr     x18
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      blr     x18
// CHECK-NOT:   The instruction is     {{[0-9a-f]+}}:      blr     x18

// CHECK-LABEL: GS-PAUTH: non-protected call found in function indirect_call_invalidates_safety, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      blr     x20
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      blr     x20
// CHECK-NOT:   The instruction is     {{[0-9a-f]+}}:      blr     x20
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        mov     x2, x0
        autiza  x2
        blr     x2              // protected call, but makes x2 unsafe
        blr     x2              // unprotected call

        mov     x8, x0
        autiza  x8
        blr     x8              // protected call, but makes x8 unsafe
        blr     x8              // unprotected call

        mov     x10, x0
        autiza  x10
        blr     x10             // protected call, but makes x10 unsafe
        blr     x10             // unprotected call

        mov     x16, x0
        autiza  x16
        blr     x16             // protected call, but makes x16 unsafe
        blr     x16             // unprotected call

        mov     x18, x0
        autiza  x18
        blr     x18             // protected call, but makes x18 unsafe
        blr     x18             // unprotected call

        mov     x20, x0
        autiza  x20
        blr     x20             // protected call, but makes x20 unsafe
        blr     x20             // unprotected call

        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size indirect_call_invalidates_safety, .-indirect_call_invalidates_safety

        .globl  direct_call_invalidates_safety_nocfg
        .type   direct_call_invalidates_safety_nocfg,@function
direct_call_invalidates_safety_nocfg:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function direct_call_invalidates_safety_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      blr     x2
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      bl      callee
// CHECK-LABEL: GS-PAUTH: non-protected call found in function direct_call_invalidates_safety_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      blr     x8
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      bl      callee
// CHECK-LABEL: GS-PAUTH: non-protected call found in function direct_call_invalidates_safety_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      blr     x10
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      bl      callee
// CHECK-LABEL: GS-PAUTH: non-protected call found in function direct_call_invalidates_safety_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      blr     x16
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      bl      callee
// CHECK-LABEL: GS-PAUTH: non-protected call found in function direct_call_invalidates_safety_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      blr     x18
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      bl      callee
// CHECK-LABEL: GS-PAUTH: non-protected call found in function direct_call_invalidates_safety_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      blr     x20
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      bl      callee
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        mov     x2, x0
        autiza  x2
        bl      callee
        blr     x2

        mov     x8, x0
        autiza  x8
        bl      callee
        blr     x8

        mov     x10, x0
        autiza  x10
        bl      callee
        blr     x10

        mov     x16, x0
        autiza  x16
        bl      callee
        blr     x16

        mov     x18, x0
        autiza  x18
        bl      callee
        blr     x18

        mov     x20, x0
        autiza  x20
        bl      callee
        blr     x20

        adr     x2, 1f
        br      x2
1:
        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size direct_call_invalidates_safety_nocfg, .-direct_call_invalidates_safety_nocfg

        .globl  indirect_call_invalidates_safety_nocfg
        .type   indirect_call_invalidates_safety_nocfg,@function
indirect_call_invalidates_safety_nocfg:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function indirect_call_invalidates_safety_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      blr     x2
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      blr     x2
// Check that only one error is reported per pair of BLRs.
// CHECK-NOT:   The instruction is     {{[0-9a-f]+}}:      blr     x2

// CHECK-LABEL: GS-PAUTH: non-protected call found in function indirect_call_invalidates_safety_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      blr     x8
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      blr     x8
// CHECK-NOT:   The instruction is     {{[0-9a-f]+}}:      blr     x8

// CHECK-LABEL: GS-PAUTH: non-protected call found in function indirect_call_invalidates_safety_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      blr     x10
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      blr     x10
// CHECK-NOT:   The instruction is     {{[0-9a-f]+}}:      blr     x10

// CHECK-LABEL: GS-PAUTH: non-protected call found in function indirect_call_invalidates_safety_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      blr     x16
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      blr     x16
// CHECK-NOT:   The instruction is     {{[0-9a-f]+}}:      blr     x16

// CHECK-LABEL: GS-PAUTH: non-protected call found in function indirect_call_invalidates_safety_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      blr     x18
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      blr     x18
// CHECK-NOT:   The instruction is     {{[0-9a-f]+}}:      blr     x18

// CHECK-LABEL: GS-PAUTH: non-protected call found in function indirect_call_invalidates_safety_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      blr     x20
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      blr     x20
// CHECK-NOT:   The instruction is     {{[0-9a-f]+}}:      blr     x20
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        mov     x2, x0
        autiza  x2
        blr     x2              // protected call, but makes x2 unsafe
        blr     x2              // unprotected call

        mov     x8, x0
        autiza  x8
        blr     x8              // protected call, but makes x8 unsafe
        blr     x8              // unprotected call

        mov     x10, x0
        autiza  x10
        blr     x10             // protected call, but makes x10 unsafe
        blr     x10             // unprotected call

        mov     x16, x0
        autiza  x16
        blr     x16             // protected call, but makes x16 unsafe
        blr     x16             // unprotected call

        mov     x18, x0
        autiza  x18
        blr     x18             // protected call, but makes x18 unsafe
        blr     x18             // unprotected call

        mov     x20, x0
        autiza  x20
        blr     x20             // protected call, but makes x20 unsafe
        blr     x20             // unprotected call

        adr     x2, 1f
        br      x2
1:
        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size indirect_call_invalidates_safety_nocfg, .-indirect_call_invalidates_safety_nocfg

// Test that fused auth+use Armv8.3 instruction do not mark register as safe.

        .globl  blraa_no_mark_safe
        .type   blraa_no_mark_safe,@function
blraa_no_mark_safe:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function blraa_no_mark_safe, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      blr     x0
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      blraa   x0, x1
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   paciasp
// CHECK-NEXT:  {{[0-9a-f]+}}:   stp     x29, x30, [sp, #-0x10]!
// CHECK-NEXT:  {{[0-9a-f]+}}:   mov     x29, sp
// CHECK-NEXT:  {{[0-9a-f]+}}:   blraa   x0, x1
// CHECK-NEXT:  {{[0-9a-f]+}}:   blr     x0
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  {{[0-9a-f]+}}:   autiasp
// CHECK-NEXT:  {{[0-9a-f]+}}:   ret
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        blraa   x0, x1  // safe, no write-back, clobbers everything
        blr     x0      // unsafe

        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size blraa_no_mark_safe, .-blraa_no_mark_safe

        .globl  blraa_no_mark_safe_nocfg
        .type   blraa_no_mark_safe_nocfg,@function
blraa_no_mark_safe_nocfg:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function blraa_no_mark_safe_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      blr     x0
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      blraa   x0, x1
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        blraa   x0, x1  // safe, no write-back, clobbers everything
        blr     x0      // detected as unsafe

        adr     x2, 1f
        br      x2
1:
        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size blraa_no_mark_safe_nocfg, .-blraa_no_mark_safe_nocfg

// Check that the correct set of registers is used to compute the set of last
// writing instructions: both x16 and x17 are tracked in this function, but
// only one particular register is used to compute the set of clobbering
// instructions in each report.

        .globl  last_insts_writing_to_reg
        .type   last_insts_writing_to_reg,@function
last_insts_writing_to_reg:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function last_insts_writing_to_reg, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:         blr     x16
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      ldr     x16, [x0]
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   paciasp
// CHECK-NEXT:  {{[0-9a-f]+}}:   stp     x29, x30, [sp, #-0x10]!
// CHECK-NEXT:  {{[0-9a-f]+}}:   mov     x29, sp
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldr     x16, [x0]
// CHECK-NEXT:  {{[0-9a-f]+}}:   blr     x16
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldr     x17, [x1]
// CHECK-NEXT:  {{[0-9a-f]+}}:   blr     x17
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  {{[0-9a-f]+}}:   autiasp
// CHECK-NEXT:  {{[0-9a-f]+}}:   ret
// CHECK-LABEL: GS-PAUTH: non-protected call found in function last_insts_writing_to_reg, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:         blr     x17
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      ldr     x17, [x1]
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   paciasp
// CHECK-NEXT:  {{[0-9a-f]+}}:   stp     x29, x30, [sp, #-0x10]!
// CHECK-NEXT:  {{[0-9a-f]+}}:   mov     x29, sp
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldr     x16, [x0]
// CHECK-NEXT:  {{[0-9a-f]+}}:   blr     x16
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldr     x17, [x1]
// CHECK-NEXT:  {{[0-9a-f]+}}:   blr     x17
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldp     x29, x30, [sp], #0x10
// CHECK-NEXT:  {{[0-9a-f]+}}:   autiasp
// CHECK-NEXT:  {{[0-9a-f]+}}:   ret
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        ldr     x16, [x0]
        blr     x16
        ldr     x17, [x1]
        blr     x17

        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size last_insts_writing_to_reg, .-last_insts_writing_to_reg

        .globl  last_insts_writing_to_reg_nocfg
        .type   last_insts_writing_to_reg_nocfg,@function
last_insts_writing_to_reg_nocfg:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function last_insts_writing_to_reg_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:         blr     x16
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      ldr     x16, [x0]
// CHECK-LABEL: GS-PAUTH: non-protected call found in function last_insts_writing_to_reg_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:         blr     x17
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      ldr     x17, [x1]
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        ldr     x16, [x0]
        blr     x16
        ldr     x17, [x1]
        blr     x17

        adr     x2, 1f
        br      x2
1:
        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size last_insts_writing_to_reg_nocfg, .-last_insts_writing_to_reg_nocfg

// Test that the instructions reported to the user are not cluttered with
// annotations attached by data-flow analysis or its CFG-unaware counterpart.

        .globl  printed_instrs_dataflow
        .type   printed_instrs_dataflow,@function
printed_instrs_dataflow:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function printed_instrs_dataflow, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:         br      x0 # TAILCALL{{ *$}}
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      ldr     x0, [x0]{{ *$}}
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   ldr     x0, [x0]{{ *$}}
// CHECK-NEXT:  {{[0-9a-f]+}}:   br      x0 # TAILCALL{{ *$}}
        ldr     x0, [x0]
        br      x0
        .size   printed_instrs_dataflow, .-printed_instrs_dataflow

        .globl  printed_instrs_nocfg
        .type   printed_instrs_nocfg,@function
printed_instrs_nocfg:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function printed_instrs_nocfg, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:         br      x0 # UNKNOWN CONTROL FLOW # Offset: 12{{ *$}}
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      ldr     x0, [x0]{{ *$}}
        adr     x2, 1f
        br      x2
1:
        ldr     x0, [x0]
        br      x0
        .size   printed_instrs_nocfg, .-printed_instrs_nocfg

// Test handling of unreachable basic blocks.
//
// Basic blocks without any predecessors were observed in real-world optimized
// code. At least sometimes they were actually reachable via jump table, which
// was not detected, but the function was processed as if its CFG was
// reconstructed successfully.
//
// As a more predictable model example, let's use really unreachable code
// for testing.

        .globl  bad_unreachable_call
        .type   bad_unreachable_call,@function
bad_unreachable_call:
// CHECK-LABEL: GS-PAUTH: Warning: possibly imprecise CFG, the analysis quality may be degraded in this function. According to BOLT, unreachable code is found in function bad_unreachable_call, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      blr     x0
// CHECK-NOT:   instructions that write to the affected registers after any authentication are:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function bad_unreachable_call, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      blr     x0
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        b       1f
        // unreachable basic block:
        blr     x0

1:      // reachable basic block:
        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size bad_unreachable_call, .-bad_unreachable_call

        .globl  good_unreachable_call
        .type   good_unreachable_call,@function
good_unreachable_call:
// CHECK-NOT: non-protected call{{.*}}good_unreachable_call
// CHECK-LABEL: GS-PAUTH: Warning: possibly imprecise CFG, the analysis quality may be degraded in this function. According to BOLT, unreachable code is found in function good_unreachable_call, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      autia   x0, x1
// CHECK-NOT: instructions that write to the affected registers after any authentication are:
// CHECK-NOT: non-protected call{{.*}}good_unreachable_call
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp

        b       1f
        // unreachable basic block:
        autia   x0, x1
        blr     x0      // <-- this call is definitely protected provided at least
                        //     basic block boundaries are detected correctly

1:      // reachable basic block:
        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size good_unreachable_call, .-good_unreachable_call

        .globl  unreachable_loop_of_bbs
        .type   unreachable_loop_of_bbs,@function
unreachable_loop_of_bbs:
// CHECK-NOT: unreachable basic blocks{{.*}}unreachable_loop_of_bbs
// CHECK-NOT: non-protected call{{.*}}unreachable_loop_of_bbs
// CHECK-LABEL: GS-PAUTH: Warning: possibly imprecise CFG, the analysis quality may be degraded in this function. According to BOLT, unreachable code is found in function unreachable_loop_of_bbs, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      blr     x0
// CHECK-NOT: unreachable basic blocks{{.*}}unreachable_loop_of_bbs
// CHECK-NOT: non-protected call{{.*}}unreachable_loop_of_bbs
        paciasp
        stp     x29, x30, [sp, #-16]!
        mov     x29, sp
        b       .Lreachable_epilogue_bb

.Lfirst_unreachable_bb:
        blr     x0      // <-- this call is not analyzed
        b       .Lsecond_unreachable_bb
.Lsecond_unreachable_bb:
        blr     x1      // <-- this call is not analyzed
        b       .Lfirst_unreachable_bb

.Lreachable_epilogue_bb:
        ldp     x29, x30, [sp], #16
        autiasp
        ret
        .size unreachable_loop_of_bbs, .-unreachable_loop_of_bbs

        .globl  main
        .type   main,@function
main:
        mov x0, 0
        ret
        .size   main, .-main
