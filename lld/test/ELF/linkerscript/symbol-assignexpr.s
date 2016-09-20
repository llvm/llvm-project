# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t

# RUN: echo "SECTIONS { \
# RUN:         symbol = CONSTANT(MAXPAGESIZE); \
# RUN:         symbol2 = symbol + 0x1234; \
# RUN:         symbol3 = symbol2; \
# RUN:         symbol4 = symbol + -4; \
# RUN:         symbol5 = symbol - ~ 0xfffb; \
# RUN:         symbol6 = symbol - ~(0xfff0 + 0xb); \
# RUN:         symbol7 = symbol - ~ 0xfffb + 4; \
# RUN:         symbol8 = ~ 0xffff + 4; \
# RUN:         symbol9 = - 4; \
# RUN:         symbol10 = 0xfedcba9876543210; \
# RUN:         symbol11 = ((0x28000 + 0x1fff) & ~(0x1000 + -1)); \
# RUN:       }" > %t.script
# RUN: ld.lld -o %t1 --script %t.script %t
# RUN: llvm-objdump -t %t1 | FileCheck %s

# CHECK:      SYMBOL TABLE:
# CHECK-NEXT: 0000000000000000 *UND* 00000000
# CHECK-NEXT:                  .text 00000000 _start
# CHECK-NEXT:                  .text 00000000 foo
# CHECK-NEXT: 0000000000200000 *ABS* 00000000 symbol
# CHECK-NEXT: 0000000000201234 *ABS* 00000000 symbol2
# CHECK-NEXT: 0000000000201234 *ABS* 00000000 symbol3
# CHECK-NEXT: 00000000001ffffc *ABS* 00000000 symbol4
# CHECK-NEXT: 000000000020fffc *ABS* 00000000 symbol5
# CHECK-NEXT: 000000000020fffc *ABS* 00000000 symbol6
# CHECK-NEXT: 0000000000210000 *ABS* 00000000 symbol7
# CHECK-NEXT: ffffffffffff0004 *ABS* 00000000 symbol8
# CHECK-NEXT: fffffffffffffffc *ABS* 00000000 symbol9
# CHECK-NEXT: fedcba9876543210 *ABS* 00000000 symbol10
# CHECK-NEXT: 0000000000029000 *ABS* 00000000 symbol11

# RUN: echo "SECTIONS { \
# RUN:         symbol2 = symbol; \
# RUN:       }" > %t2.script
# RUN: not ld.lld -o %t2 --script %t2.script %t 2>&1 \
# RUN:  | FileCheck -check-prefix=ERR %s
# ERR: symbol not found: symbol

.global _start
_start:
 nop

.global foo
foo:
