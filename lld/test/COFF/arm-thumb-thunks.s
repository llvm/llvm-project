// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=thumbv7-windows %s -o %t.obj
// RUN: lld-link -entry:main -subsystem:console %t.obj -out:%t.exe -map -verbose 2>&1 | FileCheck -check-prefix=VERBOSE %s
// RUN: llvm-objdump --no-print-imm-hex -d %t.exe --start-address=0x401000 --stop-address=0x401022 | FileCheck --check-prefix=MAIN %s
// RUN: llvm-objdump --no-print-imm-hex -d %t.exe --start-address=0x501022 --stop-address=0x501032 | FileCheck --check-prefix=FUNC1 %s
// RUN: llvm-objdump --no-print-imm-hex -d %t.exe --start-address=0x601032 | FileCheck --check-prefix=FUNC2 %s

// VERBOSE: Added 3 thunks with margin {{.*}} in 1 passes

    .syntax unified
    .globl main
    .globl func1
    .text
main:
    bne func1
    bne func2
    // This should reuse the same thunk as func1 above
    bne func1_alias
    bx lr
    .section .text$a, "xr"
    .space 0x100000
    .section .text$b, "xr"
func1:
func1_alias:
    // This shouldn't reuse the func2 thunk from above, since it is out
    // of range.
    bne func2
    bx lr
    .section .text$c, "xr"
    .space 0x100000
    .section .text$d, "xr"
func2:
// Test using string tail merging. This is irrelevant to the thunking itself,
// but running multiple passes of assignAddresses() calls finalizeAddresses()
// multiple times; check that MergeChunk handles this correctly.
    movw r0, :lower16:"??_C@string1"
    movt r0, :upper16:"??_C@string1"
    movw r1, :lower16:"??_C@string2"
    movt r1, :upper16:"??_C@string2"
    bx lr

    .section .rdata,"dr",discard,"??_C@string1"
    .globl "??_C@string1"
"??_C@string1":
    .asciz "foobar"
    .section .rdata,"dr",discard,"??_C@string2"
    .globl "??_C@string2"
"??_C@string2":
    .asciz "bar"

// MAIN:    401000:       f040 8005       bne.w   0x40100e <.text+0xe>
// MAIN:    401004:       f040 8008       bne.w   0x401018 <.text+0x18>
// MAIN:    401008:       f040 8001       bne.w   0x40100e <.text+0xe>
// MAIN:    40100c:       4770            bx      lr
// func1 thunk
// MAIN:    40100e:       f240 0c08       movw    r12, #8
// MAIN:    401012:       f2c0 0c10       movt    r12, #16
// MAIN:    401016:       44e7            add     pc,  r12
// func2 thunk
// MAIN:    401018:       f240 0c0e       movw    r12, #14
// MAIN:    40101c:       f2c0 0c20       movt    r12, #32
// MAIN:    401020:       44e7            add     pc,  r12

// FUNC1:   501022:       f040 8001       bne.w   0x501028 <.text+0x100028>
// FUNC1:   501026:       4770            bx      lr
// func2 thunk
// FUNC1:   501028:       f64f 7cfe       movw    r12, #65534
// FUNC1:   50102c:       f2c0 0c0f       movt    r12, #15
// FUNC1:   501030:       44e7            add     pc,  r12

// FUNC2:   601032:       f242 0000       movw    r0, #8192
// FUNC2:   601036:       f2c0 0060       movt    r0, #96
// FUNC2:   60103a:       f242 0103       movw    r1, #8195
// FUNC2:   60103e:       f2c0 0160       movt    r1, #96
// FUNC2:   601042:       4770            bx      lr
