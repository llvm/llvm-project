# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

# RUN: echo "SECTIONS { \
# RUN:   .aaa         : { *(.aaa) } \
# RUN:   .bbb         : { *(.bbb) } \
# RUN:   .ccc         : { *(.ccc) } \
# RUN:   _aaa = SIZEOF(.aaa); \
# RUN:   _bbb = SIZEOF(.bbb); \
# RUN:   _ccc = SIZEOF(.ccc); \
# RUN:   _ddd = SIZEOF(.not_exist); \
# RUN: }" > %t.script
# RUN: ld.lld -T %t.script %t.o -o %t
# RUN: llvm-readelf -S -s %t | FileCheck %s

# CHECK:      Name Type     Address          Off    Size
# CHECK:      .aaa PROGBITS 0000000000000000 001000 000008
# CHECK-NEXT: .bbb PROGBITS 0000000000000008 001008 000010
# CHECK-NEXT: .ccc PROGBITS 0000000000000018 001018 000018

# CHECK:        Value          Size Type   Bind   Vis     Ndx Name
# CHECK:      0000000000000008    0 NOTYPE GLOBAL DEFAULT ABS _aaa
# CHECK-NEXT: 0000000000000010    0 NOTYPE GLOBAL DEFAULT ABS _bbb
# CHECK-NEXT: 0000000000000018    0 NOTYPE GLOBAL DEFAULT ABS _ccc
## SIZEOF(.not_exist) has a value of 0.
# CHECK-NEXT: 0000000000000000    0 NOTYPE GLOBAL DEFAULT ABS _ddd

.global _start
_start:
 nop

.section .aaa,"a"
 .quad 0

.section .bbb,"a"
 .quad 0
 .quad 0

.section .ccc,"a"
 .quad 0
 .quad 0
 .quad 0
