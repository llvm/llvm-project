// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64-windows %s -o %t.obj
// RUN: lld-link -entry:main -subsystem:console %t.obj -out:%t.exe -map -verbose 2>&1 | FileCheck -check-prefix=VERBOSE %s
// RUN: llvm-objdump --no-print-imm-hex -d %t.exe | FileCheck --check-prefix=DISASM %s

// VERBOSE: Added 2 thunks with margin {{.*}} in 1 passes

    .globl main
    .globl func1
    .globl func2
    .text
main:
    tbz w0, #0, func1
    ret
    .section .text$a, "xr"
    .space 0x8000
    .section .text$b, "xr"
func1:
    tbz w0, #0, func2
    ret
    .space 1
    .section .text$c, "xr"
    .space 0x8000
    .section .text$d, "xr"
    .align 2
func2:
    ret

// DISASM: 0000000140001000 <.text>:
// DISASM: 140001000:      36000040        tbz     w0, #0, 0x140001008 <.text+0x8>
// DISASM: 140001004:      d65f03c0        ret
// DISASM: 140001008:      90000050        adrp    x16, 0x140009000
// DISASM: 14000100c:      91005210        add     x16, x16, #20
// DISASM: 140001010:      d61f0200        br      x16

// DISASM: 140009014:      36000060        tbz     w0, #0, 0x140009020 <.text+0x8020>
// DISASM: 140009018:      d65f03c0        ret

// DISASM: 140009020:      90000050        adrp    x16, 0x140011000
// DISASM: 140009024:      9100b210        add     x16, x16, #44
// DISASM: 140009028:      d61f0200        br      x16

// DISASM: 14001102c:      d65f03c0        ret
