// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
// RUN: ld.lld --fix-cortex-a53-843419 -Ttext=0x8000000 %t.o -o %t2
// RUN: llvm-objdump -d --start-address=0x8001000 --stop-address=0x8001004 %t2 | FileCheck %s
/// The temporary files are more than 128MiB. Delete them early.
// RUN: rm %t.o %t2

.section .text.01, "ax", %progbits
.balign 4096
.space 4096 - 8
adrp x0, thunk
ldr x1, [x1, #0]
// CHECK: <thunk>:
// CHECK-NEXT: b 0xc001008 <__CortexA53843419_8001000>
thunk:
ldr x0, [x0, :got_lo12:thunk]
ret
.space 64 * 1024 * 1024

.section .text.02, "ax", %progbits
.space 64 * 1024 * 1024
