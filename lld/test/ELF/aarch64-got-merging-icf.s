// REQUIRES: aarch64

# RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t
# RUN: ld.lld %t -o %t2 --icf=all
# RUN: llvm-objdump --section-headers %t2 | FileCheck %s --check-prefix=EXE

# RUN: ld.lld -shared %t -o %t3 --icf=all
# RUN: llvm-objdump --section-headers %t3 | FileCheck %s --check-prefix=DSO

## All .rodata.* sections should merge into a single GOT entry
# EXE: {{.*}}.got 00000008{{.*}}

## When symbols are preemptible in DSO mode, GOT entries wouldn't be merged
# DSO: {{.*}}.got 00000020{{.*}}

.addrsig

callee:
ret

.macro f, index, isglobal

# (Kept unique) first instruction of the GOT code sequence
.section .text.f1_\index,"ax",@progbits
f1_\index:
adrp x0, :got:g\index
mov x1, #\index
b f2_\index

# Folded, second instruction of the GOT code sequence
.section .text.f2_\index,"ax",@progbits
f2_\index:
ldr x0, [x0, :got_lo12:g\index] 
b callee

# Folded
.ifnb \isglobal
.globl g\index
.endif
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

f 0 1
f 1 1
f 2 1
f 3