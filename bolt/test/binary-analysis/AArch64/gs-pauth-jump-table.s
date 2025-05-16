// -Wl,--no-relax prevents converting ADRP+ADD pairs into NOP+ADR.
// Without -Wl,--emit-relocs BOLT refuses to create CFG information for the below functions.

// RUN: %clang %cflags -march=armv8.3-a -Wl,--no-relax -Wl,--emit-relocs %s -o %t.exe
// RUN: llvm-bolt-binary-analysis --scanners=pauth                         %t.exe 2>&1 | FileCheck --check-prefixes=CHECK,CFG %s
// RUN: llvm-bolt-binary-analysis --scanners=pauth --auth-traps-on-failure %t.exe 2>&1 | FileCheck --check-prefixes=CHECK,CFG %s
// RUN: %clang %cflags -march=armv8.3-a -Wl,--no-relax %s -o %t.exe
// RUN: llvm-bolt-binary-analysis --scanners=pauth                         %t.exe 2>&1 | FileCheck --check-prefixes=CHECK,NOCFG %s
// RUN: llvm-bolt-binary-analysis --scanners=pauth --auth-traps-on-failure %t.exe 2>&1 | FileCheck --check-prefixes=CHECK,NOCFG %s

// FIXME: Labels could be further validated. Specifically, it could be checked
//        that the jump table itself is located in a read-only data section.

// FIXME: BOLT does not reconstruct CFG correctly for jump tables yet, thus
//        register state is pessimistically reset to unsafe at the beginning of
//        each basic block without any predecessors.
//        Until CFG reconstruction is fixed, add paciasp+autiasp instructions to
//        silence "non-protected ret" false-positives and explicitly ignore
//        "Warning: the function has unreachable basic blocks..." lines.

        .text
        .p2align 2
        .globl  good_jump_table
        .type   good_jump_table,@function
good_jump_table:
// CHECK-NOT: good_jump_table
// CFG:       GS-PAUTH: Warning: the function has unreachable basic blocks (possibly incomplete CFG) in function good_jump_table
// CHECK-NOT: good_jump_table
        paciasp
        cmp     x16, #0x2
        csel    x16, x16, xzr, ls
        adrp    x17, 4f
        add     x17, x17, :lo12:4f
        ldrsw   x16, [x17, x16, lsl #2]
1:
        adr     x17, 1b
        add     x16, x17, x16
        br      x16
2:
        autiasp
        ret
3:
        autiasp
        ret
        .size good_jump_table, .-good_jump_table
        .section .rodata,"a",@progbits
        .p2align 2, 0x0
4:
        .word   2b-1b
        .word   3b-1b

// NOP (HINT #0) before ADR is correct (it can be produced by linker due to
// relaxing ADRP+ADD sequence), but other HINT instructions are not.

        .text
        .p2align 2
        .globl  jump_table_relaxed_adrp_add
        .type   jump_table_relaxed_adrp_add,@function
jump_table_relaxed_adrp_add:
// CHECK-NOT: jump_table_relaxed_adrp_add
// CFG:       GS-PAUTH: Warning: the function has unreachable basic blocks (possibly incomplete CFG) in function jump_table_relaxed_adrp_add
// CHECK-NOT: jump_table_relaxed_adrp_add
        paciasp
        cmp     x16, #0x2
        csel    x16, x16, xzr, ls
        hint    #0                 // nop
        adr     x17, 4f
        ldrsw   x16, [x17, x16, lsl #2]
1:
        adr     x17, 1b
        add     x16, x17, x16
        br      x16
2:
        autiasp
        ret
3:
        autiasp
        ret
        .size jump_table_relaxed_adrp_add, .-jump_table_relaxed_adrp_add
        .section .rodata,"a",@progbits
        .p2align 2, 0x0
4:
        .word   2b-1b
        .word   3b-1b

        .text
        .p2align 2
        .globl  jump_table_wrong_hint
        .type   jump_table_wrong_hint,@function
jump_table_wrong_hint:
// CFG-LABEL:   GS-PAUTH: non-protected call found in function jump_table_wrong_hint, basic block {{[^,]+}}, at address
// NOCFG-LABEL: GS-PAUTH: non-protected call found in function jump_table_wrong_hint, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      br      x16 # UNKNOWN CONTROL FLOW
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      add     x16, x17, x16
// CFG-NEXT:    This happens in the following basic block:
// CFG-NEXT:    {{[0-9a-f]+}}:   adr     x17, __ENTRY_jump_table_wrong_hint@0x{{[0-9a-f]+}}
// CFG-NEXT:    {{[0-9a-f]+}}:   add     x16, x17, x16
// CFG-NEXT:    {{[0-9a-f]+}}:   br      x16 # UNKNOWN CONTROL FLOW
        paciasp
        cmp     x16, #0x2
        csel    x16, x16, xzr, ls
        hint    #20                // unknown hint
        adr     x17, 4f
        ldrsw   x16, [x17, x16, lsl #2]
1:
        adr     x17, 1b
        add     x16, x17, x16
        br      x16
2:
        autiasp
        ret
3:
        autiasp
        ret
        .size jump_table_wrong_hint, .-jump_table_wrong_hint
        .section .rodata,"a",@progbits
        .p2align 2, 0x0
4:
        .word   2b-1b
        .word   3b-1b

// For now, all registers are permitted as temporary ones, not only x16 and x17.

        .text
        .p2align 2
        .globl  jump_table_unsafe_reg_1
        .type   jump_table_unsafe_reg_1,@function
jump_table_unsafe_reg_1:
// CHECK-NOT: jump_table_unsafe_reg_1
// CFG:       GS-PAUTH: Warning: the function has unreachable basic blocks (possibly incomplete CFG) in function jump_table_unsafe_reg_1
// CHECK-NOT: jump_table_unsafe_reg_1
        paciasp
        cmp     x1, #0x2
        csel    x1, x1, xzr, ls
        adrp    x17, 4f
        add     x17, x17, :lo12:4f
        ldrsw   x1, [x17, x1, lsl #2]
1:
        adr     x17, 1b
        add     x1, x17, x1
        br      x1
2:
        autiasp
        ret
3:
        autiasp
        ret
        .size jump_table_unsafe_reg_1, .-jump_table_unsafe_reg_1
        .section .rodata,"a",@progbits
        .p2align 2, 0x0
4:
        .word   2b-1b
        .word   3b-1b

        .text
        .p2align 2
        .globl  jump_table_unsafe_reg_2
        .type   jump_table_unsafe_reg_2,@function
jump_table_unsafe_reg_2:
// CHECK-NOT: jump_table_unsafe_reg_2
// CFG:       GS-PAUTH: Warning: the function has unreachable basic blocks (possibly incomplete CFG) in function jump_table_unsafe_reg_2
// CHECK-NOT: jump_table_unsafe_reg_2
        paciasp
        cmp     x16, #0x2
        csel    x16, x16, xzr, ls
        adrp    x1, 4f
        add     x1, x1, :lo12:4f
        ldrsw   x16, [x1, x16, lsl #2]
1:
        adr     x1, 1b
        add     x16, x1, x16
        br      x16
2:
        autiasp
        ret
3:
        autiasp
        ret
        .size jump_table_unsafe_reg_2, .-jump_table_unsafe_reg_2
        .section .rodata,"a",@progbits
        .p2align 2, 0x0
4:
        .word   2b-1b
        .word   3b-1b

// FIXME: Detect possibility of jump table overflow.
        .text
        .p2align 2
        .globl  jump_table_wrong_limit
        .type   jump_table_wrong_limit,@function
jump_table_wrong_limit:
// CHECK-NOT: jump_table_wrong_limit
// CFG:       GS-PAUTH: Warning: the function has unreachable basic blocks (possibly incomplete CFG) in function jump_table_wrong_limit
// CHECK-NOT: jump_table_wrong_limit
        paciasp
        cmp     x16, #0x1000
        csel    x16, x16, xzr, ls
        adrp    x17, 4f
        add     x17, x17, :lo12:4f
        ldrsw   x16, [x17, x16, lsl #2]
1:
        adr     x17, 1b
        add     x16, x17, x16
        br      x16
2:
        autiasp
        ret
3:
        autiasp
        ret
        .size jump_table_wrong_limit, .-jump_table_wrong_limit
        .section .rodata,"a",@progbits
        .p2align 2, 0x0
4:
        .word   2b-1b
        .word   3b-1b

        .text
        .p2align 2
        .globl  jump_table_unrelated_inst_1
        .type   jump_table_unrelated_inst_1,@function
jump_table_unrelated_inst_1:
// CFG-LABEL:   GS-PAUTH: non-protected call found in function jump_table_unrelated_inst_1, basic block {{[^,]+}}, at address
// NOCFG-LABEL: GS-PAUTH: non-protected call found in function jump_table_unrelated_inst_1, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      br      x16 # UNKNOWN CONTROL FLOW
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      add     x16, x17, x16
// CFG-NEXT:    This happens in the following basic block:
// CFG-NEXT:    {{[0-9a-f]+}}:   adr     x17, __ENTRY_jump_table_unrelated_inst_1@0x{{[0-9a-f]+}}
// CFG-NEXT:    {{[0-9a-f]+}}:   nop
// CFG-NEXT:    {{[0-9a-f]+}}:   add     x16, x17, x16
// CFG-NEXT:    {{[0-9a-f]+}}:   br      x16 # UNKNOWN CONTROL FLOW
        paciasp
        cmp     x16, #0x2
        csel    x16, x16, xzr, ls
        adrp    x17, 4f
        add     x17, x17, :lo12:4f
        ldrsw   x16, [x17, x16, lsl #2]
1:
        adr     x17, 1b
        nop
        add     x16, x17, x16
        br      x16
2:
        autiasp
        ret
3:
        autiasp
        ret
        .size jump_table_unrelated_inst_1, .-jump_table_unrelated_inst_1
        .section .rodata,"a",@progbits
        .p2align 2, 0x0
4:
        .word   2b-1b
        .word   3b-1b

        .text
        .p2align 2
        .globl  jump_table_unrelated_inst_2
        .type   jump_table_unrelated_inst_2,@function
jump_table_unrelated_inst_2:
// CFG-LABEL:   GS-PAUTH: non-protected call found in function jump_table_unrelated_inst_2, basic block {{[^,]+}}, at address
// NOCFG-LABEL: GS-PAUTH: non-protected call found in function jump_table_unrelated_inst_2, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      br      x16 # UNKNOWN CONTROL FLOW
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      add     x16, x17, x16
// CFG-NEXT:    This happens in the following basic block:
// CFG-NEXT:    {{[0-9a-f]+}}:   adr     x17, __ENTRY_jump_table_unrelated_inst_2@0x{{[0-9a-f]+}}
// CFG-NEXT:    {{[0-9a-f]+}}:   add     x16, x17, x16
// CFG-NEXT:    {{[0-9a-f]+}}:   br      x16 # UNKNOWN CONTROL FLOW
        paciasp
        cmp     x16, #0x2
        csel    x16, x16, xzr, ls
        adrp    x17, 4f
        add     x17, x17, :lo12:4f
        nop
        ldrsw   x16, [x17, x16, lsl #2]
1:
        adr     x17, 1b
        add     x16, x17, x16
        br      x16
2:
        autiasp
        ret
3:
        autiasp
        ret
        .size jump_table_unrelated_inst_2, .-jump_table_unrelated_inst_2
        .section .rodata,"a",@progbits
        .p2align 2, 0x0
4:
        .word   2b-1b
        .word   3b-1b

        .text
        .p2align 2
        .globl  jump_table_multiple_predecessors_1
        .type   jump_table_multiple_predecessors_1,@function
jump_table_multiple_predecessors_1:
// NOCFG-NOT:   jump_table_multiple_predecessors_1
// CFG-LABEL:   GS-PAUTH: non-protected call found in function jump_table_multiple_predecessors_1, basic block {{[^,]+}}, at address
// CFG-NEXT:    The instruction is     {{[0-9a-f]+}}:      br      x16 # UNKNOWN CONTROL FLOW
// CFG-NEXT:    The 1 instructions that write to the affected registers after any authentication are:
// CFG-NEXT:    1.     {{[0-9a-f]+}}:      add     x16, x17, x16
// CFG-NEXT:    This happens in the following basic block:
// CFG-NEXT:    {{[0-9a-f]+}}:   adr     x17, __ENTRY_jump_table_multiple_predecessors_1@0x{{[0-9a-f]+}}
// CFG-NEXT:    {{[0-9a-f]+}}:   add     x16, x17, x16
// CFG-NEXT:    {{[0-9a-f]+}}:   br      x16 # UNKNOWN CONTROL FLOW
        paciasp
        cbz     x1, 1f          // this instruction can jump to the middle of the sequence
        cmp     x16, #0x2
        csel    x16, x16, xzr, ls
        adrp    x17, 4f
        add     x17, x17, :lo12:4f
        ldrsw   x16, [x17, x16, lsl #2]
1:
        adr     x17, 1b         // multiple predecessors are possible
        add     x16, x17, x16
        br      x16
2:
        autiasp
        ret
3:
        autiasp
        ret
        .size jump_table_multiple_predecessors_1, .-jump_table_multiple_predecessors_1
        .section .rodata,"a",@progbits
        .p2align 2, 0x0
4:
        .word   2b-1b
        .word   3b-1b

        .text
        .p2align 2
        .globl  jump_table_multiple_predecessors_2
        .type   jump_table_multiple_predecessors_2,@function
jump_table_multiple_predecessors_2:
// NOCFG-NOT:   jump_table_multiple_predecessors_2
// CFG-LABEL:   GS-PAUTH: non-protected call found in function jump_table_multiple_predecessors_2, basic block {{[^,]+}}, at address
// CFG-NEXT:    The instruction is     {{[0-9a-f]+}}:      br      x16 # UNKNOWN CONTROL FLOW
// CFG-NEXT:    The 1 instructions that write to the affected registers after any authentication are:
// CFG-NEXT:    1.     {{[0-9a-f]+}}:      add     x16, x17, x16
// CFG-NEXT:    This happens in the following basic block:
// CFG-NEXT:    {{[0-9a-f]+}}:   adr     x17, __ENTRY_jump_table_multiple_predecessors_2@0x{{[0-9a-f]+}}
// CFG-NEXT:    {{[0-9a-f]+}}:   add     x16, x17, x16
// CFG-NEXT:    {{[0-9a-f]+}}:   br      x16 # UNKNOWN CONTROL FLOW
        paciasp
        cbz     x1, 5f              // this instruction can jump to the middle of the sequence
        cmp     x16, #0x2
        csel    x16, x16, xzr, ls
5:
        adrp    x17, 4f             // multiple predecessors are possible
        add     x17, x17, :lo12:4f
        ldrsw   x16, [x17, x16, lsl #2]
1:
        adr     x17, 1b
        add     x16, x17, x16
        br      x16
2:
        autiasp
        ret
3:
        autiasp
        ret
        .size jump_table_multiple_predecessors_2, .-jump_table_multiple_predecessors_2
        .section .rodata,"a",@progbits
        .p2align 2, 0x0
4:
        .word   2b-1b
        .word   3b-1b

// Test a few pattern violations...

        .text
        .p2align 2
        .globl  jump_table_wrong_reg_1
        .type   jump_table_wrong_reg_1,@function
jump_table_wrong_reg_1:
// CFG-LABEL:   GS-PAUTH: non-protected call found in function jump_table_wrong_reg_1, basic block {{[^,]+}}, at address
// NOCFG-LABEL: GS-PAUTH: non-protected call found in function jump_table_wrong_reg_1, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      br      x1 # UNKNOWN CONTROL FLOW
// CHECK-NEXT:  The 0 instructions that write to the affected registers after any authentication are:
        paciasp
        cmp     x16, #0x2
        csel    x16, x16, xzr, ls
        adrp    x17, 4f
        add     x17, x17, :lo12:4f
        ldrsw   x16, [x17, x16, lsl #2]
1:
        adr     x17, 1b
        add     x16, x17, x16
        br      x1             // wrong reg
2:
        autiasp
        ret
3:
        autiasp
        ret
        .size jump_table_wrong_reg_1, .-jump_table_wrong_reg_1
        .section .rodata,"a",@progbits
        .p2align 2, 0x0
4:
        .word   2b-1b
        .word   3b-1b

        .text
        .p2align 2
        .globl  jump_table_wrong_reg_2
        .type   jump_table_wrong_reg_2,@function
jump_table_wrong_reg_2:
// CFG-LABEL:   GS-PAUTH: non-protected call found in function jump_table_wrong_reg_2, basic block {{[^,]+}}, at address
// NOCFG-LABEL: GS-PAUTH: non-protected call found in function jump_table_wrong_reg_2, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      br      x16 # UNKNOWN CONTROL FLOW
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      add     x16, x17, x1
// CFG-NEXT:    This happens in the following basic block:
// CFG-NEXT:    {{[0-9a-f]+}}:   adr     x17, __ENTRY_jump_table_wrong_reg_2@0x{{[0-9a-f]+}}
// CFG-NEXT:    {{[0-9a-f]+}}:   add     x16, x17, x1
// CFG-NEXT:    {{[0-9a-f]+}}:   br      x16 # UNKNOWN CONTROL FLOW
        paciasp
        cmp     x16, #0x2
        csel    x16, x16, xzr, ls
        adrp    x17, 4f
        add     x17, x17, :lo12:4f
        ldrsw   x16, [x17, x16, lsl #2]
1:
        adr     x17, 1b
        add     x16, x17, x1  // wrong reg
        br      x16
2:
        autiasp
        ret
3:
        autiasp
        ret
        .size jump_table_wrong_reg_2, .-jump_table_wrong_reg_2
        .section .rodata,"a",@progbits
        .p2align 2, 0x0
4:
        .word   2b-1b
        .word   3b-1b

        .text
        .p2align 2
        .globl  jump_table_wrong_reg_3
        .type   jump_table_wrong_reg_3,@function
jump_table_wrong_reg_3:
// CFG-LABEL:   GS-PAUTH: non-protected call found in function jump_table_wrong_reg_3, basic block {{[^,]+}}, at address
// NOCFG-LABEL: GS-PAUTH: non-protected call found in function jump_table_wrong_reg_3, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      br      x16 # UNKNOWN CONTROL FLOW
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      add     x16, x17, x16
// CFG-NEXT:    This happens in the following basic block:
// CFG-NEXT:    {{[0-9a-f]+}}:   adr     x17, __ENTRY_jump_table_wrong_reg_3@0x{{[0-9a-f]+}}
// CFG-NEXT:    {{[0-9a-f]+}}:   add     x16, x17, x16
// CFG-NEXT:    {{[0-9a-f]+}}:   br      x16 # UNKNOWN CONTROL FLOW
        paciasp
        cmp     x16, #0x2
        csel    x16, x16, xzr, ls
        adrp    x17, 4f
        add     x17, x1, :lo12:4f        // wrong reg
        ldrsw   x16, [x17, x16, lsl #2]
1:
        adr     x17, 1b
        add     x16, x17, x16
        br      x16
2:
        autiasp
        ret
3:
        autiasp
        ret
        .size jump_table_wrong_reg_3, .-jump_table_wrong_reg_3
        .section .rodata,"a",@progbits
        .p2align 2, 0x0
4:
        .word   2b-1b
        .word   3b-1b

        .text
        .p2align 2
        .globl  jump_table_wrong_reg_4
        .type   jump_table_wrong_reg_4,@function
jump_table_wrong_reg_4:
// CFG-LABEL:   GS-PAUTH: non-protected call found in function jump_table_wrong_reg_4, basic block {{[^,]+}}, at address
// NOCFG-LABEL: GS-PAUTH: non-protected call found in function jump_table_wrong_reg_4, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      br      x16 # UNKNOWN CONTROL FLOW
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      add     x16, x17, x16
// CFG-NEXT:    This happens in the following basic block:
// CFG-NEXT:    {{[0-9a-f]+}}:   adr     x17, __ENTRY_jump_table_wrong_reg_4@0x{{[0-9a-f]+}}
// CFG-NEXT:    {{[0-9a-f]+}}:   add     x16, x17, x16
// CFG-NEXT:    {{[0-9a-f]+}}:   br      x16 # UNKNOWN CONTROL FLOW
        paciasp
        cmp     x16, #0x2
        csel    x16, x16, x1, ls  // wrong reg
        adrp    x17, 4f
        add     x17, x17, :lo12:4f
        ldrsw   x16, [x17, x16, lsl #2]
1:
        adr     x17, 1b
        add     x16, x17, x16
        br      x16
2:
        autiasp
        ret
3:
        autiasp
        ret
        .size jump_table_wrong_reg_4, .-jump_table_wrong_reg_4
        .section .rodata,"a",@progbits
        .p2align 2, 0x0
4:
        .word   2b-1b
        .word   3b-1b

        .text
        .p2align 2
        .globl  jump_table_wrong_imm_1
        .type   jump_table_wrong_imm_1,@function
jump_table_wrong_imm_1:
// CFG-LABEL:   GS-PAUTH: non-protected call found in function jump_table_wrong_imm_1, basic block {{[^,]+}}, at address
// NOCFG-LABEL: GS-PAUTH: non-protected call found in function jump_table_wrong_imm_1, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      br      x16 # UNKNOWN CONTROL FLOW
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      add     x16, x17, x16
// CFG-NEXT:    This happens in the following basic block:
// CFG-NEXT:    {{[0-9a-f]+}}:   adr     x17, __ENTRY_jump_table_wrong_imm_1@0x{{[0-9a-f]+}}
// CFG-NEXT:    {{[0-9a-f]+}}:   add     x16, x17, x16
// CFG-NEXT:    {{[0-9a-f]+}}:   br      x16 # UNKNOWN CONTROL FLOW
        paciasp
        cmp     x16, #0x2
        csel    x16, x16, xzr, ls
        adrp    x17, 4f
        add     x17, x17, :lo12:4f
        ldrsw   x16, [x17, x16, sxtx #2]  // wrong: sxtx instead of lsl
1:
        adr     x17, 1b
        add     x16, x17, x16
        br      x16
2:
        autiasp
        ret
3:
        autiasp
        ret
        .size jump_table_wrong_imm_1, .-jump_table_wrong_imm_1
        .section .rodata,"a",@progbits
        .p2align 2, 0x0
4:
        .word   2b-1b
        .word   3b-1b

        .text
        .p2align 2
        .globl  jump_table_wrong_imm_2
        .type   jump_table_wrong_imm_2,@function
jump_table_wrong_imm_2:
// CFG-LABEL:   GS-PAUTH: non-protected call found in function jump_table_wrong_imm_2, basic block {{[^,]+}}, at address
// NOCFG-LABEL: GS-PAUTH: non-protected call found in function jump_table_wrong_imm_2, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      br      x16 # UNKNOWN CONTROL FLOW
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      add     x16, x17, x16
// CFG-NEXT:    This happens in the following basic block:
// CFG-NEXT:    {{[0-9a-f]+}}:   adr     x17, __ENTRY_jump_table_wrong_imm_2@0x{{[0-9a-f]+}}
// CFG-NEXT:    {{[0-9a-f]+}}:   add     x16, x17, x16
// CFG-NEXT:    {{[0-9a-f]+}}:   br      x16 # UNKNOWN CONTROL FLOW
        paciasp
        cmp     x16, #0x2
        csel    x16, x16, xzr, lt  // wrong: lt instead of ls
        adrp    x17, 4f
        add     x17, x17, :lo12:4f
        ldrsw   x16, [x17, x16, lsl #2]
1:
        adr     x17, 1b
        add     x16, x17, x16
        br      x16
2:
        autiasp
        ret
3:
        autiasp
        ret
        .size jump_table_wrong_imm_2, .-jump_table_wrong_imm_2
        .section .rodata,"a",@progbits
        .p2align 2, 0x0
4:
        .word   2b-1b
        .word   3b-1b

        .text
        .p2align 2
        .globl  jump_table_wrong_imm_3
        .type   jump_table_wrong_imm_3,@function
jump_table_wrong_imm_3:
// CFG-LABEL:   GS-PAUTH: non-protected call found in function jump_table_wrong_imm_3, basic block {{[^,]+}}, at address
// NOCFG-LABEL: GS-PAUTH: non-protected call found in function jump_table_wrong_imm_3, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      br      x16 # UNKNOWN CONTROL FLOW
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      add     x16, x17, x16
// CFG-NEXT:    This happens in the following basic block:
// CFG-NEXT:    {{[0-9a-f]+}}:   adr     x17, __ENTRY_jump_table_wrong_imm_3@0x{{[0-9a-f]+}}
// CFG-NEXT:    {{[0-9a-f]+}}:   add     x16, x17, x16
// CFG-NEXT:    {{[0-9a-f]+}}:   br      x16 # UNKNOWN CONTROL FLOW
        paciasp
        cmp     x16, #0x2
        csel    x16, x16, xzr, ls
        adrp    x17, 4f
        add     x17, x17, :lo12:4f
        ldrsw   x16, [x17, x16, lsl #2]
1:
        adr     x17, 1b
        add     x16, x17, x16, lsl #2  // wrong: lsl #2
        br      x16
2:
        autiasp
        ret
3:
        autiasp
        ret
        .size jump_table_wrong_imm_3, .-jump_table_wrong_imm_3
        .section .rodata,"a",@progbits
        .p2align 2, 0x0
4:
        .word   2b-1b
        .word   3b-1b

// CFI instructions should be skipped and should not prevent matching
// the instruction sequence.

        .text
        .p2align 2
        .globl  skip_cfi_instructions
        .type   skip_cfi_instructions,@function
skip_cfi_instructions:
        .cfi_startproc
// CHECK-NOT: skip_cfi_instructions
// CFG:       GS-PAUTH: Warning: the function has unreachable basic blocks (possibly incomplete CFG) in function skip_cfi_instructions
// CHECK-NOT: skip_cfi_instructions
        paciasp
        cmp     x16, #0x2
        csel    x16, x16, xzr, ls
        adrp    x17, 4f
        .cfi_def_cfa_offset 16      // should be skipped over when matching the sequence
        add     x17, x17, :lo12:4f
        ldrsw   x16, [x17, x16, lsl #2]
1:
        adr     x17, 1b
        add     x16, x17, x16
        br      x16
2:
        autiasp
        ret
3:
        autiasp
        ret
        .size skip_cfi_instructions, .-skip_cfi_instructions
        .cfi_endproc
        .section .rodata,"a",@progbits
        .p2align 2, 0x0
4:
        .word   2b-1b
        .word   3b-1b

        .text
        .p2align 2
        .globl  incomplete_jump_table
        .type   incomplete_jump_table,@function
incomplete_jump_table:
// CFG-LABEL:   GS-PAUTH: non-protected call found in function incomplete_jump_table, basic block {{[^,]+}}, at address
// NOCFG-LABEL: GS-PAUTH: non-protected call found in function incomplete_jump_table, at address
// CHECK-NEXT:  The instruction is     {{[0-9a-f]+}}:      br      x16 # UNKNOWN CONTROL FLOW
// CHECK-NEXT:  The 1 instructions that write to the affected registers after any authentication are:
// CHECK-NEXT:  1.     {{[0-9a-f]+}}:      add     x16, x17, x16
// CFG-NEXT:    This happens in the following basic block:
// CFG-NEXT:    {{[0-9a-f]+}}:   adr     x17, __ENTRY_incomplete_jump_table@0x{{[0-9a-f]+}}
// CFG-NEXT:    {{[0-9a-f]+}}:   add     x16, x17, x16
// CFG-NEXT:    {{[0-9a-f]+}}:   br      x16 # UNKNOWN CONTROL FLOW
        // Do not try to step past the start of the function.
        ldrsw   x16, [x17, x16, lsl #2]
1:
        adr     x17, 1b
        add     x16, x17, x16
        br      x16
2:
        autiasp
        ret
3:
        autiasp
        ret
        .size incomplete_jump_table, .-incomplete_jump_table
        .section .rodata,"a",@progbits
        .p2align 2, 0x0
4:
        .word   2b-1b
        .word   3b-1b

        .text
        .globl  main
        .type   main,@function
main:
        mov x0, 0
        ret
        .size   main, .-main
