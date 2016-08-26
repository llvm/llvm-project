# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t

# RUN: echo "SECTIONS { \
# RUN:         symbol = CONSTANT(MAXPAGESIZE); \
# RUN:         symbol2 = symbol + 0x1234; \
# RUN:         symbol3 = symbol2; \
# RUN:       }" > %t.script
# RUN: ld.lld -o %t1 --script %t.script %t
# RUN: llvm-objdump -t %t1 | FileCheck %s

# CHECK:      SYMBOL TABLE:
# CHECK-NEXT: 0000000000000000 *UND* 00000000
# CHECK-NEXT: 0000000000000120 .text 00000000 _start
# CHECK-NEXT: 0000000000000121 .text 00000000 foo
# CHECK-NEXT: 0000000000200000 *ABS* 00000000 symbol
# CHECK-NEXT: 0000000000201234 *ABS* 00000000 symbol2
# CHECK-NEXT: 0000000000201234 *ABS* 00000000 symbol3

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
