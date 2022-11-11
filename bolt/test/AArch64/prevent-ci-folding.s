// This test checks that functions containing Constant Islands are not folded even
// if they have the same data

// RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
// RUN: %clang %cflags --target=aarch64-unknown-linux %t.o -o %t.exe -Wl,-q
// RUN: llvm-bolt -icf -lite=false %t.exe -o %t.bolt
// RUN: llvm-objdump -d -j .text %t.bolt | FileCheck %s

// CHECK: <func1>:
// CHECK: <func2>:

func1:
    add x0, x0, #1
    ret
    .word 0xdeadbeef
    .word 0xdeadbeef
.size func1, .-func1

func2:
    add x0, x0, #1
    ret
    .word 0xdeadbeef
    .word 0xdeadbeef
.size func2, .-func2

.global        main
.type  main, %function
main:
    mov x0, #0
    bl     func1
    bl     func2
    sub     x0, x0, #2
    mov     w8, #93
    svc     #0
