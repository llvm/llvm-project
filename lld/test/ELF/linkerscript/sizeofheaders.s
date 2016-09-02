# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: echo " SECTIONS {             \
# RUN:          . = SIZEOF_HEADERS;  \
# RUN:          _size = SIZEOF_HEADERS;  \
# RUN:          .text : {*(.text*)} \
# RUN:          }" > %t.script
# RUN: ld.lld -o %t1 --script %t.script %t
# RUN: llvm-objdump -t %t1 | FileCheck %s

#CHECK:      SYMBOL TABLE:
#CHECK-NEXT:  0000000000000000 *UND* 00000000
#CHECK-NEXT:  0000000000000120 .text 00000000 _start
#CHECK-NEXT:  0000000000000120 *ABS* 00000000 _size

.global _start
_start:
 nop
