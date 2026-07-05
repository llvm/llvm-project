// REQUIRES: aarch64

// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows %s -o %t-arm64ec.obj
// RUN: llvm-mc -filetype=obj -triple=aarch64-windows %s -o %t-arm64.obj
// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows %S/Inputs/loadconfig-arm64ec.s -o %t-loadcfg.obj

// Check that .wowthk section is merged into .text on ARM64EC target.

// RUN: lld-link -out:%t.dll -machine:arm64ec %t-arm64ec.obj %t-loadcfg.obj -dll -noentry
// RUN: llvm-objdump -d %t.dll | FileCheck  -check-prefix=DISASM %s
// DISASM:      0000000180001000 <.text>:
// DISASM-NEXT: 180001000: 52800040     mov     w0, #0x2                // =2
// DISASM-NEXT: 180001004: d65f03c0     ret
// DISASM-NEXT: 180001008: 52800060     mov     w0, #0x3                // =3
// DISASM-NEXT: 18000100c: d65f03c0     ret

// Check that .wowthk section is not merged on aarch64 target.

// RUN: lld-link -out:%t.dll -machine:arm64 %t-arm64.obj -dll -noentry
// RUN: llvm-objdump -d %t.dll | FileCheck -check-prefix=DISASM2 %s
// DISASM2:      0000000180001000 <.text>:
// DISASM2-NEXT: 180001000: 52800040     mov     w0, #0x2                // =2
// DISASM2-NEXT: 180001004: d65f03c0     ret
// DISASM2-EMPTY:
// DISASM2-NEXT: Disassembly of section .wowthk:
// DISASM2-EMPTY:
// DISASM2-NEXT: 0000000180002000 <.wowthk>:
// DISASM2-NEXT: 180002000: 52800060     mov     w0, #0x3                // =3
// DISASM2-NEXT: 180002004: d65f03c0     ret


        .text
        .globl arm64ec_func_sym
        .p2align 2, 0x0
arm64ec_func_sym:
        mov w0, #2
        ret

        .section .wowthk$aa, "x"
        .globl wowthk_sym
        .p2align 3, 0x0
wowthk_sym:
        mov w0, #3
        ret
