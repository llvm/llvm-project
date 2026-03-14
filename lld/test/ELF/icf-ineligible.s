# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o /dev/null --keep-unique fu --icf=all --print-icf-sections | FileCheck %s

## Check that ICF is able to merge equivalent sections with relocations to
## different symbols, e.g. aliases, that refer to the same section which is
## ineligible for ICF.

# CHECK: selected section {{.*}}:(.text.f1)
# CHECK:   removing identical section {{.*}}:(.text.f2)
# CHECK:   removing identical section {{.*}}:(.text.f3)
# CHECK: selected section {{.*}}:(.text.f4)
# CHECK:   removing identical section {{.*}}:(.text.f5)

.globl d, d_alias, fu, f1, f2, f3, f4, f5

.section .data.d,"aw",@progbits
d:
d_alias:
.long 42

.section .text.fu,"ax",@progbits
fu:
  nop

.section .text.f1,"ax",@progbits
f1:
.quad d

.section .text.f2,"ax",@progbits
f2:
.quad d_alias

.section .text.f3,"ax",@progbits
f3:
.quad .data.d

.section .text.f4,"ax",@progbits
f4:
.quad fu

.section .text.f5,"ax",@progbits
f5:
.quad .text.fu
