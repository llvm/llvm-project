# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t

## Check that data-in-code information is sorted even if
## sections are reordered compared to the input order.

# RUN: sed -e s/SYM/_first/ %t/input.s | \
# RUN:   llvm-mc -filetype=obj --triple=x86_64-apple-darwin -o %t/first.o
# RUN: sed -e s/SYM/_second/ %t/input.s | \
# RUN:   llvm-mc -filetype=obj --triple=x86_64-apple-darwin -o %t/second.o
# RUN: sed -e s/SYM/_third/ %t/input.s | \
# RUN:   llvm-mc -filetype=obj --triple=x86_64-apple-darwin -o %t/third.o
# RUN: %lld -dylib -lSystem -order_file %t/order.txt %t/first.o %t/second.o %t/third.o -o %t/out
# RUN: llvm-objdump --macho --syms %t/out > %t/dump.txt
# RUN: llvm-objdump --macho --data-in-code %t/out >> %t/dump.txt
# RUN: FileCheck %s < %t/dump.txt

# CHECK-LABEL: SYMBOL TABLE:
# CHECK-DAG:   [[#%x, SECOND:]] g     F __TEXT,__text _second
# CHECK-DAG:   [[#%x, FIRST:]]  g     F __TEXT,__text _first
# CHECK-DAG:   [[#%x, THIRD:]]  g     F __TEXT,__text _third

# CHECK-LABEL: Data in code table (3 entries)
# CHECK-NEXT:  offset              length kind
# CHECK-NEXT:  0x[[#%.8x, SECOND]] 4      JUMP_TABLE32
# CHECK-NEXT:  0x[[#%.8x, FIRST]]  4      JUMP_TABLE32
# CHECK-NEXT:  0x[[#%.8x, THIRD]]  4      JUMP_TABLE32

#--- order.txt
_second
_first
_third

#--- input.s
.globl SYM
SYM:
.data_region jt32
.long 0
.end_data_region
