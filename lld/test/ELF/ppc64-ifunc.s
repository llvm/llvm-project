# REQUIRES: ppc

# RUN: llvm-mc -filetype=obj -triple=powerpc64le-unknown-linux %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=powerpc64le-unknown-linux %p/Inputs/shared-ppc64.s -o %t2.o
# RUN: ld.lld -shared %t2.o -o %t2.so
# RUN: ld.lld %t.o %t2.so -o %t
# RUN: llvm-objdump -D %t | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux %p/Inputs/shared-ppc64.s -o %t2.o
# RUN: ld.lld -shared %t2.o -o %t2.so
# RUN: ld.lld %t.o %t2.so -o %t
# RUN: llvm-objdump -D %t | FileCheck %s

# CHECK: Disassembly of section .text:

# Tocbase    + (-2 << 16) + 32576
# 0x100380d0 + (-131072)  + 32576 = 0x10020010 (.got.plt[2])
# CHECK: __plt_foo:
# CHECK-NEXT:     std 2, 24(1)
# CHECK-NEXT:     addis 12, 2, -2
# CHECK-NEXT:     ld 12, 32576(12)
# CHECK-NEXT:     mtctr 12
# CHECK-NEXT:     bctr

# Tocbase    + (-2 << 16) + 32584
# 0x100380d0 + (-131072)  + 32584 = 0x10020018  (.got.plt[3])
# CHECK: __plt_ifunc:
# CHECK-NEXT:     std 2, 24(1)
# CHECK-NEXT:     addis 12, 2, -2
# CHECK-NEXT:     ld 12, 32584(12)
# CHECK-NEXT:     mtctr 12
# CHECK-NEXT:     bctr

# CHECK: ifunc:
# CHECK-NEXT: 10010028:  {{.*}} nop

# CHECK: _start:
# CHECK-NEXT:     addis 2, 12, 3
# CHECK-NEXT:     addi 2, 2, -32604
# CHECK-NEXT:     bl .+67108812
# CHECK-NEXT:     ld 2, 24(1)
# CHECK-NEXT:     bl .+67108824
# CHECK-NEXT:     ld 2, 24(1)

# Address of .got.plt
# CHECK:      Disassembly of section .got.plt:
# CHECK-NEXT:   .got.plt:
# CHECK-NEXT:   10020000:


# Check tocbase
# CHECK:       Disassembly of section .got:
# CHECK-NEXT:    .got:
# CHECK-NEXT:    100300d0:


    .text
    .abiversion 2

.type ifunc STT_GNU_IFUNC
.globl ifunc
ifunc:
 nop

    .global _start
    .type   _start,@function

_start:
.Lfunc_gep0:
  addis 2, 12, .TOC.-.Lfunc_gep0@ha
  addi 2, 2, .TOC.-.Lfunc_gep0@l
.Lfunc_lep0:
  .localentry     _start, .Lfunc_lep0-.Lfunc_gep0
  bl foo
  nop
  bl ifunc
  nop
