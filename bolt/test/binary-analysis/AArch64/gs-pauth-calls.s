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

        .globl  main
        .type   main,@function
main:
        mov x0, 0
        ret
        .size   main, .-main
