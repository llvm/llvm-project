// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=thumbv7-windows %s -o %t.obj
// RUN: lld-link -entry:main -subsystem:console %t.obj -out:%t.exe -verbose 2>&1 | FileCheck -check-prefix=VERBOSE %s
// RUN: llvm-objdump --no-print-imm-hex -d %t.exe --start-address=0x403000 --stop-address=0x403008 | FileCheck --check-prefix=FUNC01 %s
// RUN: llvm-objdump --no-print-imm-hex -d %t.exe --start-address=0x404ffa --stop-address=0x405012 | FileCheck --check-prefix=FUNC01-THUNKS %s

// VERBOSE: Added {{.*}} thunks with margin 204800 in 2 passes

    .syntax unified
    .globl main
    .text
main:
    b  func01
    bx lr

.irp i, 01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12, 13, 14, 15, 16, 17, 18
    .section .text$\i\()a, "xr"
    .balign 8192
func\i:
    bne far_func\i
    bne func_within_margin\i
    // Originally, the first section is less than 8192 bytes large, and the
    // second one follows almost directly. After adding one thunk after
    // the first section, the second one will move forward by 8192 bytes
    // due to the alignment.
    .space 8192 - 8 - 4

    .section .text$\i\()b, "xr"
    .balign 8192
align\i:
    nop
.endr

    .section .text$999, "xr"
tail:
    .space 0x100000 - 100*1024 - 18*8192*2
    // Initially, these symbols are within range from all the sections above,
    // even when taking the initial margin into account. After adding thunks
    // to all the sections above, some of these are also out of range, forcing
    // running a second pass.
.irp i, 01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12, 13, 14, 15, 16, 17, 18
func_within_margin\i:
    nop
.endr
    .space 0x100000

    // These are always out of range.
.irp i, 01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11, 12, 13, 14, 15, 16, 17, 18
far_func\i:
    nop
.endr
    bx lr

// FUNC01: 403000:       f041 87fc       bne.w   0x404ffc <.text+0x3ffc>
// FUNC01: 403004:       f041 87ff       bne.w   0x405006 <.text+0x4006>

// Check that we only have two thunks here, even if we created the first
// thunk twice (once in the first pass, then thrown away and recreated
// in the second pass).

// FUNC01-THUNKS: 404ffa:       0000            movs    r0,  r0
// The instruction above is padding from the .space
// FUNC01-THUNKS: 404ffc:       f247 0c1e       movw    r12, #28702
// FUNC01-THUNKS: 405000:       f2c0 0c20       movt    r12, #32
// FUNC01-THUNKS: 405004:       44e7            add     pc,  r12
// FUNC01-THUNKS: 405006:       f646 7cf0       movw    r12, #28656
// FUNC01-THUNKS: 40500a:       f2c0 0c10       movt    r12, #16
// FUNC01-THUNKS: 40500e:       44e7            add     pc, r12
// The instruction below is padding from the .balign
// FUNC01-THUNKS: 405010:       cccc            ldm     r4!, {r2, r3, r6, r7}
