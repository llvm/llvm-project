# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo " SECTIONS {             \
# RUN:          . = SIZEOF_HEADERS;  \
# RUN:          _size = SIZEOF_HEADERS;  \
# RUN:          .text : {*(.text*)} \
# RUN:          }" > %t.script
# RUN: ld.lld -o %t --script %t.script %t.o
# RUN: llvm-readelf -s %t | FileCheck %s

# CHECK:         Value         Size Type   Bind   Vis     Ndx Name
# CHECK:      0000000000000120    0 NOTYPE GLOBAL DEFAULT   1 _start
# CHECK-NEXT: 0000000000000120    0 NOTYPE GLOBAL DEFAULT ABS _size

.global _start
_start:
 nop
