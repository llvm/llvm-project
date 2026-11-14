// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64-windows %s -o %t.obj
// RUN: lld-link -entry:main -subsystem:console %t.obj -out:%t.exe -verbose 2>&1 | FileCheck -check-prefix=VERBOSE %s
// RUN: llvm-objdump --no-print-imm-hex -d %t.exe --start-address=0x140001000 --stop-address=0x140001008 | FileCheck %s

// The margin is wider than the range of a TBZ, so without this every TBZ got a
// thunk. The one to near, in its own chunk, stays direct; the one to far gets
// the one thunk. near is a function so that the assembler keeps the relocation.

// VERBOSE: Added 1 thunks with margin {{.*}} in 1 passes

// CHECK:      140001000: {{.*}}tbz w0, #0, 0x14000100c
// CHECK-NEXT: 140001004: {{.*}}tbz w0, #0, 0x140001010

    .globl main
    .globl near
    .text
main:
    tbz w0, #0, near
    tbz w0, #0, far
    ret
    .def near; .scl 2; .type 32; .endef
near:
    ret
    .section .text$a, "xr"
    .space 0x8000
    .section .text$b, "xr"
    .globl far
far:
    ret
