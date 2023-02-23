# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=i686 %s -o %t.o
# RUN: ld.lld --icf=all %t.o -o /dev/null --print-icf-sections 2>&1 | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-freebsd %s -o %t.o
# RUN: ld.lld --icf=all %t.o -o /dev/null --print-icf-sections 2>&1 | FileCheck %s

# Checks that ICF does not merge 2 sections the offset of
# the relocations of which differ.

# CHECK-NOT: selected

.section .text.orig,"ax"
  .quad -1

.section .text.foo,"ax"
  .quad -1
  .reloc 0, BFD_RELOC_NONE, 0

.section .text.bar,"ax"
  .quad -1
  .reloc 1, BFD_RELOC_NONE, 0
