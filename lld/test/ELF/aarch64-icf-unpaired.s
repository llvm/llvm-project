// REQUIRES: aarch64

# RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t
# RUN: ld.lld %t -o %t2 --icf=all --print-icf-sections | FileCheck %s

# CHECK-NOT: selected section {{.*}}(.text.f2_0)
# CHECK-NOT:   removing identical section {{.*}}(.text.f2_1)
# CHECK-NOT:   removing identical section {{.*}}(.text.f2_2)

.addrsig

callee:
ret

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

.globl g\index
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
f 2