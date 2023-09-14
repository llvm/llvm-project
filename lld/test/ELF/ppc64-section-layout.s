# REQUIRES: ppc
## Test PPC64 specific section layout. See also section-layout.s.

# RUN: llvm-mc -filetype=obj -triple=powerpc64le %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=powerpc64le %p/Inputs/shared.s -o %t1.o
# RUN: ld.lld -shared -soname=t1.so %t1.o -o %t1.so
# RUN: ld.lld %t.o %t1.so -o %t
# RUN: llvm-readelf -S %t | FileCheck %s

# CHECK:      .text
# CHECK-NEXT: .tdata
# CHECK-NEXT: .tbss
# CHECK-NEXT: .dynamic
# CHECK-NEXT: .got
# CHECK-NEXT: .toc
## The end of .relro_padding is aligned to a common-page-size boundary.
# CHECK-NEXT: .relro_padding NOBITS 0000000010020400 000400 000c00 00 WA 0 0 1
# CHECK-NEXT: .data
# CHECK-NEXT: .branch_lt

.globl _start
_start:
  addis 3, 2, bar2@got@ha
  ld    3, bar2@got@l(3)
  addis 5, 2, .Lbar@toc@ha
  ld    5, .Lbar@toc@l(5)

.section .toc,"aw",@progbits
.Lbar:
  .tc bar[TC], bar

.section .tdata,"awT",@progbits; .space 1
.section .tbss,"awT",@nobits; .space 1
.section .data,"aw",@progbits; .space 1
