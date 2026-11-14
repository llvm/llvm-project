// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=thumbv7-windows %s -o %t.obj
// RUN: lld-link -entry:main -subsystem:console %t.obj -out:%t.exe -verbose 2>&1 | FileCheck -check-prefix=VERBOSE %s
// RUN: llvm-objdump --no-print-imm-hex -d %t.exe --start-address=0x401000 --stop-address=0x401008 | FileCheck --check-prefix=MAIN %s

// A branch to a target in its own section chunk gets no thunk while it is in
// range, even within the margin, because its distance can't change. A thunk
// would clobber r12. far_target is out of range and gets the one thunk.
// local_target is a function so that the assembler keeps the relocation.

// VERBOSE: Added 1 thunks with margin {{.*}} in 1 passes

// MAIN: 401000: {{.*}}bne.w 0x4f1004 <.text+0xf0004>
// MAIN-NEXT: 401004: {{.*}}bne.w 0x4f1006 <.text+0xf0006>

    .syntax unified
    .globl main
    .text
    .def main; .scl 2; .type 32; .endef
main:
    bne.w local_target
    bne.w far_target
    bx lr
    .space 0xF0000 - 6
    .def local_target; .scl 3; .type 32; .endef
local_target:
    bx lr
    .section .text$z, "xr"
    .space 0x100000
    .def far_target; .scl 2; .type 32; .endef
    .globl far_target
far_target:
    bx lr
