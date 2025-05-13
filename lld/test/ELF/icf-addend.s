# REQUIRES: x86
# RUN: rm -rf %t && split-file %s %t && cd %t

#--- trivial-relocation.s
# For trivial relocations, merging two equivalent sections is allowed but we must not
# merge their symbols if addends are different.

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux trivial-relocation.s -o trivial.o
# RUN: ld.lld trivial.o -o /dev/null --icf=all --print-icf-sections | FileCheck %s

# CHECK: selected section {{.*}}:(.text.f1)
# CHECK:   removing identical section {{.*}}:(.text.f2)
# CHECK-NOT: redirecting 'y' in symtab to x
# CHECK-NOT: redirecting 'y' to 'x'

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

#--- non-trivial-relocation.s
# For non-trivial relocations, we must not merge sections if addends are different.
# Not merging sections would automatically disable symbol merging.

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux trivial-relocation.s -o trivial.o
# RUN: ld.lld trivial.o -o /dev/null --icf=all --print-icf-sections | FileCheck %s

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
movq x+4@GOTPCREL(%rip), %rax

.section .text.f2,"ax",@progbits
f2:
movq y@GOTPCREL(%rip), %rax

.section .text._start,"ax",@progbits
.globl _start
_start:
call f1
call f2

