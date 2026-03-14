# REQUIRES: x86

## Test we enable data sections for LTO so --symbol-ordering-fils is respected
## for data symbols.

# RUN: llvm-mc -filetype=obj -triple=x86_64-scei-ps4 %s -o %t.o
# RUN: llvm-as %p/Inputs/data-ordering-lto.ll -o %t.bc

# RUN: echo "tin  " > %t_order_lto.txt
# RUN: echo "dipsy " >> %t_order_lto.txt
# RUN: echo "pat " >> %t_order_lto.txt

# RUN: ld.lld --symbol-ordering-file %t_order_lto.txt %t.o %t.bc -o %t
# RUN: llvm-nm -v %t | FileCheck %s

# CHECK:      D tin
# CHECK-NEXT: D dipsy
# CHECK-NEXT: D pat

.globl _start
_start:
  movl $pat, %ecx
  movl $dipsy, %ebx
  movl $tin, %eax
