# REQUIRES: x86
## COMDAT groups are deduplicated by the name of the signature symbol.
## The local/global status is not part of the equation.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld %t.o %t.o -o %t
# RUN: llvm-readelf -s -x .zero -x .comdat %t | FileCheck %s

# CHECK:      Type   Bind  Vis     Ndx     Name
# CHECK-NEXT: NOTYPE LOCAL DEFAULT UND
# CHECK-NEXT: NOTYPE LOCAL DEFAULT [[#A:]] zero
# CHECK-NEXT: NOTYPE LOCAL DEFAULT [[#]]   comdat
# CHECK-NEXT: NOTYPE LOCAL DEFAULT [[#A]]  zero
# CHECK-NOT:  {{.}}

## Non-GRP_COMDAT groups are never deduplicated.
# CHECK:      Hex dump of section '.zero':
# CHECK-NEXT: [[#%x,]] 0202

## GRP_COMDAT groups are deduplicated.
# CHECK:      Hex dump of section '.comdat':
# CHECK-NEXT: [[#%x,]] 01 .{{$}}

.section .zero,"aG",@progbits,zero
zero:
  .byte 2

.section .comdat,"aG",@progbits,comdat,comdat
comdat:
  .byte 1
