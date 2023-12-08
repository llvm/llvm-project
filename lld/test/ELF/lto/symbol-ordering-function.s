# REQUIRES: x86

## Test we enable function sections for LTO so --symbol-ordering-fils is respected
## for function symbols.

# RUN: llvm-mc -filetype=obj -triple=x86_64-scei-ps4 %s -o %t.o
# RUN: llvm-as %p/Inputs/symbol-ordering-lto.ll -o %t.bc

# RUN: echo "tin  " > %t_order_lto.txt
# RUN: echo "_start " >> %t_order_lto.txt
# RUN: echo "pat " >> %t_order_lto.txt

# RUN: ld.lld --symbol-ordering-file %t_order_lto.txt %t.o %t.bc -o %t
# RUN: llvm-nm -v %t | FileCheck %s

# CHECK:      T tin
# CHECK-NEXT: T _start
# CHECK-NEXT: T pat

.globl _start
_start:
  call pat
  call tin
