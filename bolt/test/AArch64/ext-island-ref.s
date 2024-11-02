// This test checks that references to the middle of CI from other function
// are handled properly

// RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
// RUN: %clang %cflags --target=aarch64-unknown-linux %t.o -o %t.exe -Wl,-q
// RUN: llvm-bolt %t.exe -o %t.bolt -lite=false
// RUN: llvm-objdump -d -j .text %t.bolt | FileCheck %s

// CHECK: deadbeef
// CHECK-NEXT: deadbeef
// CHECK-NEXT: [[#%x,ADDR:]]: deadbeef
// CHECK-NEXT: deadbeef
// CHECK: <func>:
// CHECK-NEXT: ldr	x0, 0x[[#ADDR]]
// CHECK-NEXT: ret

.type	funcWithIsland, %function
funcWithIsland:
    ret
    .word 0xdeadbeef
    .word 0xdeadbeef
    .word 0xdeadbeef
    .word 0xdeadbeef
.size   funcWithIsland, .-funcWithIsland

.type	func, %function
func:
    ldr x0, funcWithIsland + 12
    ret
.size   func, .-func

.global	main
.type	main, %function
main:
    bl funcWithIsland
    bl func
    mov     w8, #93
    svc     #0
.size   main, .-main
