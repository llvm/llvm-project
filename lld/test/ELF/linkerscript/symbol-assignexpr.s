# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

# RUN: echo "SECTIONS { \
# RUN:         symbol = CONSTANT(MAXPAGESIZE); \
# RUN:         symbol2 = symbol + 0x1234; \
# RUN:         symbol3 = symbol2; \
# RUN:         symbol4 = symbol + -4; \
# RUN:         symbol5 = symbol - ~0xfffb; \
# RUN:         symbol6 = symbol - ~(0xfff0 + 0xb); \
# RUN:         symbol7 = symbol - ~ 0xfffb + 4; \
# RUN:         symbol8 = ~ 0xffff + 4; \
# RUN:         symbol9 = - 4; \
# RUN:         symbol10 = 0xfedcba9876543210; \
# RUN:         symbol11 = ((0x28000 + 0x1fff) & ~(0x1000 + -1)); \
# RUN:         symbol12 = 0x1234; \
# RUN:         symbol12 += 1; \
# RUN:         symbol13 = !1; \
# RUN:         symbol14 = !0; \
# RUN:         symbol15 = 0!=1; \
# RUN:         bar = 0x5678; \
# RUN:         baz = 0x9abc; \
# RUN:       }" > %t.script
# RUN: ld.lld -o %t -T %t.script %t.o
# RUN: llvm-nm -p %t | FileCheck %s

# CHECK:      0000000000000000 T _start
# CHECK-NEXT: 0000000000005678 A bar
# CHECK-NEXT: 0000000000009abc A baz
# CHECK-NEXT: 0000000000000001 T foo
# CHECK-NEXT: 0000000000001000 A symbol
# CHECK-NEXT: 0000000000002234 A symbol2
# CHECK-NEXT: 0000000000002234 A symbol3
# CHECK-NEXT: 0000000000000ffc A symbol4
# CHECK-NEXT: 0000000000010ffc A symbol5
# CHECK-NEXT: 0000000000010ffc A symbol6
# CHECK-NEXT: 0000000000011000 A symbol7
# CHECK-NEXT: ffffffffffff0004 A symbol8
# CHECK-NEXT: fffffffffffffffc A symbol9
# CHECK-NEXT: fedcba9876543210 A symbol10
# CHECK-NEXT: 0000000000029000 A symbol11
# CHECK-NEXT: 0000000000001235 A symbol12
# CHECK-NEXT: 0000000000000000 A symbol13
# CHECK-NEXT: 0000000000000001 A symbol14
# CHECK-NEXT: 0000000000000001 A symbol15

# RUN: echo "SECTIONS { symbol2 = symbol; }" > %t2.script
# RUN: not ld.lld -o /dev/null -T %t2.script %t.o 2>&1 \
# RUN:  | FileCheck -check-prefix=ERR %s
# ERR: {{.*}}.script:1: symbol not found: symbol

.global _start
_start:
 nop

.global foo
foo:
 nop

.global bar
bar = 0x1234

.comm baz,8,8
