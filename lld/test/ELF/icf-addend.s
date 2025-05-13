# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o /dev/null --icf=all --print-icf-sections | FileCheck --allow-empty %s

# Check that ICF doesn't fold sections containing functions that references
# unmergeable symbols. We should only merge symbols of two relocations when
# their addends are same.

# CHECK-NOT: selected section {{.*}}:(.text.f1)
# CHECK-NOT:   removing identical section {{.*}}:(.text.f2)

.globl x, y

.section .rodata,"a",@progbits
x:
.long 11
y:
.long 12

.section .text.f1,"ax",@progbits
f1:
movq x+4(%rip), %rax 

.section .text.f2,"ax",@progbits
f2:
movq y(%rip), %rax

.section .text._start,"ax",@progbits
.globl _start
_start:
call f1
call f2
