// REQUIRES: aarch64
// RUN: split-file %s %t.dir && cd %t.dir

// RUN: llvm-mc -filetype=obj -triple=arm64ec-windows func-arm64ec.s -o func-arm64ec.obj
// RUN: llvm-mc -filetype=obj -triple=aarch64-windows func-arm64.s -o func-arm64.obj
// RUN: lld-link -machine:arm64x -dll -noentry -out:out.dll func-arm64ec.obj func-arm64.obj
// RUN: llvm-objdump -d out.dll | FileCheck %s

// CHECK:      0000000180001000 <.text>:
// CHECK-NEXT: 180001000: 52800020     mov     w0, #0x1                // =1
// CHECK-NEXT: 180001004: d65f03c0     ret
// CHECK-NEXT:                 ...
// CHECK-NEXT: 180002000: 52800020     mov     w0, #0x1                // =1
// CHECK-NEXT: 180002004: d65f03c0     ret


#--- func-arm64.s
        .section .text,"xr",discard,func
        .globl func
        .p2align 2
func:
        mov w0, #1
        ret

        .data
        .rva func

#--- func-arm64ec.s
        .section .text,"xr",discard,"#func"
        .globl "#func"
        .p2align 2
"#func":
        mov w0, #1
        ret

        .data
        .rva "#func"
