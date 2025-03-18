// REQUIRES: aarch64

// RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t
// RUN: ld.lld %t -o %t2 --icf=all
// RUN: llvm-objdump --section-headers %t2 | FileCheck %s

// CHECK: {{.*}}.got 00000008{{.*}}

.addrsig

callee:
ret

.section .rodata.dummy1,"a",@progbits
sym1:
.long 111
.long 122
.byte 123

.section .rodata.dummy2,"a",@progbits
sym2:
.long 111
.long 122
sym3:
.byte 123

.macro f, index

.section .text.f1_\index,"ax",@progbits
f1_\index:
adrp x0, :got:g\index
mov x1, #\index
b f2_\index

.section .text.f2_\index,"ax",@progbits
f2_\index:
ldr x0, [x0, :got_lo12:g\index] 
b callee

.section .rodata.g\index,"a",@progbits
g_\index:
.long 111
.long 122

g\index:
.byte 123

.section .text._start,"ax",@progbits
bl f1_\index

.endm

.section .text._start,"ax",@progbits
.globl _start
_start:

f 0
f 1