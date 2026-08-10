# REQUIRES: x86

## Confirm --dynamic-list identifies symbols by entries, including wildcards.
## Entries need not match a symbol. 

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

# RUN: echo '{ [fb]o?1*; };' > %t.list
# RUN: ld.lld -pie --dynamic-list %t.list %t.o -o %t
# RUN: llvm-readelf --dyn-syms %t | FileCheck %s

# CHECK:      Symbol table '.dynsym' contains 4 entries:
# CHECK:      boo1
# CHECK-NEXT: foo1
# CHECK-NEXT: foo11

.globl _start, boo1, foo1, foo11, foo2
_start:
foo1:
foo11:
foo2:
boo1:
