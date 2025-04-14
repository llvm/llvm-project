// -Wl,--no-relax prevents converting ADRP+ADD pairs into NOP+ADR.
// RUN: %clang %cflags -march=armv8.3-a -Wl,--no-relax %s -o %t.exe
// RUN: llvm-bolt-binary-analysis --scanners=pauth %t.exe 2>&1 | FileCheck %s

// Test various patterns that should or should not be considered safe
// materialization of PC-relative addresses.
//
// Note that while "instructions that write to the affected registers"
// section of the report is still technically correct, it does not necessarily
// mention the instructions that are used incorrectly.
//
// FIXME: Switch to PAC* instructions instead of indirect tail call for testing
//        if a register is considered safe when detection of signing oracles is
//        implemented, as it is more traditional usage of PC-relative constants.
//        Moreover, using PAC instructions would improve test robustness, as
//        handling of *calls* can be influenced by what BOLT classifies as a
//        tail call, for example.

        .text

// Define a function that is reachable by ADR instruction.
        .type   sym,@function
sym:
        ret
        .size   sym, .-sym

        .globl  good_adr
        .type   good_adr,@function
good_adr:
// CHECK-NOT: good_adr
        adr     x0, sym
        br      x0
        .size   good_adr, .-good_adr

        .globl  good_adrp
        .type   good_adrp,@function
good_adrp:
// CHECK-NOT: good_adrp
        adrp    x0, sym
        br      x0
        .size   good_adrp, .-good_adrp

        .globl  good_adrp_add
        .type   good_adrp_add,@function
good_adrp_add:
// CHECK-NOT: good_adrp_add
        adrp    x0, sym
        add     x0, x0, :lo12:sym
        br      x0
        .size   good_adrp_add, .-good_adrp_add

        .globl  good_adrp_add_with_const_offset
        .type   good_adrp_add_with_const_offset,@function
good_adrp_add_with_const_offset:
// CHECK-NOT: good_adrp_add_with_const_offset
        adrp    x0, sym
        add     x0, x0, :lo12:sym
        add     x0, x0, #8
        br      x0
        .size   good_adrp_add_with_const_offset, .-good_adrp_add_with_const_offset

        .globl  bad_adrp_with_nonconst_offset
        .type   bad_adrp_with_nonconst_offset,@function
bad_adrp_with_nonconst_offset:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function bad_adrp_with_nonconst_offset, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      br      x0 # TAILCALL
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      add     x0, x0, x1
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   adrp    x0, #{{.*}}
// CHECK-NEXT:  {{[0-9a-f]+}}:   add     x0, x0, x1
// CHECK-NEXT:  {{[0-9a-f]+}}:   br      x0 # TAILCALL
        adrp    x0, sym
        add     x0, x0, x1
        br      x0
        .size   bad_adrp_with_nonconst_offset, .-bad_adrp_with_nonconst_offset

        .globl  bad_split_adrp
        .type   bad_split_adrp,@function
bad_split_adrp:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function bad_split_adrp, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      br      x0 # UNKNOWN CONTROL FLOW
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      add     x0, x0, #0x{{[0-9a-f]+}}
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   add     x0, x0, #0x{{[0-9a-f]+}}
// CHECK-NEXT:  {{[0-9a-f]+}}:   br      x0 # UNKNOWN CONTROL FLOW
        cbz     x2, 1f
        adrp    x0, sym
1:
        add     x0, x0, :lo12:sym
        br      x0
        .size   bad_split_adrp, .-bad_split_adrp

// Materialization of absolute addresses is not handled, as it is not expected
// to be used by real-world code, but can be supported if needed.

        .globl  bad_immediate_constant
        .type   bad_immediate_constant,@function
bad_immediate_constant:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function bad_immediate_constant, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      br      x0 # TAILCALL
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      mov     x0, #{{.*}}
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   mov     x0, #{{.*}}
// CHECK-NEXT:  {{[0-9a-f]+}}:   br      x0 # TAILCALL
        movz    x0, #1234
        br      x0
        .size   bad_immediate_constant, .-bad_immediate_constant

// Any ADR or ADRP instruction followed by any number of increments/decrements
// by constant is considered safe.

        .globl  good_adr_with_add
        .type   good_adr_with_add,@function
good_adr_with_add:
// CHECK-NOT: good_adr_with_add
        adr     x0, sym
        add     x0, x0, :lo12:sym
        br      x0
        .size   good_adr_with_add, .-good_adr_with_add

        .globl  good_adrp_with_add_non_consecutive
        .type   good_adrp_with_add_non_consecutive,@function
good_adrp_with_add_non_consecutive:
// CHECK-NOT: good_adrp_with_add_non_consecutive
        adrp    x0, sym
        mul     x1, x2, x3
        add     x0, x0, :lo12:sym
        br      x0
        .size   good_adrp_with_add_non_consecutive, .-good_adrp_with_add_non_consecutive

        .globl  good_many_offsets
        .type   good_many_offsets,@function
good_many_offsets:
// CHECK-NOT: good_many_offsets
        adrp    x0, sym
        add     x1, x0, #8
        add     x2, x1, :lo12:sym
        br      x2
        .size   good_many_offsets, .-good_many_offsets

        .globl  good_negative_offset
        .type   good_negative_offset,@function
good_negative_offset:
// CHECK-NOT: good_negative_offset
        adr     x0, sym
        sub     x1, x0, #8
        br      x1
        .size   good_negative_offset, .-good_negative_offset

// MOV Xd, Xm (which is an alias of ORR Xd, XZR, Xm) is handled as part of
// support for address arithmetics, but ORR in general is not.
// This restriction may be relaxed in the future.

        .globl  good_mov_reg
        .type   good_mov_reg,@function
good_mov_reg:
// CHECK-NOT: good_mov_reg
        adrp    x0, sym
        mov     x1, x0
        orr     x2, xzr, x1 // the same as "mov x2, x1"
        br      x2
        .size   good_mov_reg, .-good_mov_reg

        .globl  bad_orr_not_xzr
        .type   bad_orr_not_xzr,@function
bad_orr_not_xzr:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function bad_orr_not_xzr, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      br      x2 # TAILCALL
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      orr     x2, x1, x0
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   adrp    x0, #{{(0x)?[0-9a-f]+}}
// CHECK-NEXT:  {{[0-9a-f]+}}:   mov     x1, #0
// CHECK-NEXT:  {{[0-9a-f]+}}:   orr     x2, x1, x0
// CHECK-NEXT:  {{[0-9a-f]+}}:   br      x2 # TAILCALL
        adrp    x0, sym
        // The generic case of "orr Xd, Xn, Xm" is not allowed so far,
        // even if Xn is known to be safe
        movz    x1, #0
        orr     x2, x1, x0
        br      x2
        .size   bad_orr_not_xzr, .-bad_orr_not_xzr

        .globl  bad_orr_not_lsl0
        .type   bad_orr_not_lsl0,@function
bad_orr_not_lsl0:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function bad_orr_not_lsl0, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      br      x2 # TAILCALL
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      orr     x2, xzr, x0, lsl #1
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   adrp    x0, #{{(0x)?[0-9a-f]+}}
// CHECK-NEXT:  {{[0-9a-f]+}}:   orr     x2, xzr, x0, lsl #1
// CHECK-NEXT:  {{[0-9a-f]+}}:   br      x2 # TAILCALL
        adrp    x0, sym
        // Currently, the only allowed form of "orr" is that used by "mov Xd, Xn" alias.
        // This can be relaxed in the future.
        orr     x2, xzr, x0, lsl #1
        br      x2
        .size   bad_orr_not_lsl0, .-bad_orr_not_lsl0

// Check that the input register operands of `add`/`mov` is correct.

        .globl  bad_add_input_reg
        .type   bad_add_input_reg,@function
bad_add_input_reg:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function bad_add_input_reg, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      br      x0 # TAILCALL
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      add     x0, x1, #0x{{[0-9a-f]+}}
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   adrp    x0, #{{(0x)?[0-9a-f]+}}
// CHECK-NEXT:  {{[0-9a-f]+}}:   add     x0, x1, #0x{{[0-9a-f]+}}
// CHECK-NEXT:  {{[0-9a-f]+}}:   br      x0 # TAILCALL
        adrp    x0, sym
        add     x0, x1, :lo12:sym
        br      x0
        .size   bad_add_input_reg, .-bad_add_input_reg

        .globl  bad_mov_input_reg
        .type   bad_mov_input_reg,@function
bad_mov_input_reg:
// CHECK-LABEL: GS-PAUTH: non-protected call found in function bad_mov_input_reg, basic block {{[^,]+}}, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      br      x0 # TAILCALL
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      mov     x0, x1
// CHECK-NEXT:  This happens in the following basic block:
// CHECK-NEXT:  {{[0-9a-f]+}}:   adrp    x0, #{{(0x)?[0-9a-f]+}}
// CHECK-NEXT:  {{[0-9a-f]+}}:   mov     x0, x1
// CHECK-NEXT:  {{[0-9a-f]+}}:   br      x0 # TAILCALL
        adrp    x0, sym
        mov     x0, x1
        br      x0
        .size   bad_mov_input_reg, .-bad_mov_input_reg

        .globl  main
        .type   main,@function
main:
        mov     x0, 0
        ret
        .size   main, .-main
