# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: echo "SECTIONS { \
# RUN:  . = 1000h;              \
# RUN:  .hex1 : { *(.hex.1) }   \
# RUN:  . = 1010H;              \
# RUN:  .hex2 : { *(.hex.2) }   \
# RUN:  . = 10k;                \
# RUN:  .kilo1 : { *(.kilo.1) } \
# RUN:  . = 11K;                \
# RUN:  .kilo2 : { *(.kilo.2) } \
# RUN:  . = 1m;                 \
# RUN:  .mega1 : { *(.mega.1) } \
# RUN:  . = 2M;                 \
# RUN:  .mega2 : { *(.mega.2) } \
# RUN: }" > %t.script
# RUN: ld.lld %t --script %t.script -o %t2
# RUN: llvm-objdump -section-headers %t2 | FileCheck %s

# CHECK:     Sections:
# CHECK-NEXT: Idx Name          Size      Address        
# CHECK-NEXT:   0               00000000 0000000000000000
# CHECK-NEXT:   1 .hex1         00000008 0000000000001000
# CHECK-NEXT:   2 .hex2         00000008 0000000000001010
# CHECK-NEXT:   3 .kilo1        00000008 0000000000002800
# CHECK-NEXT:   4 .kilo2        00000008 0000000000002c00
# CHECK-NEXT:   5 .mega1        00000008 0000000000100000
# CHECK-NEXT:   6 .mega2        00000008 0000000000200000

## Mailformed number errors.
# RUN: echo "SECTIONS { \
# RUN:  . = 0x11h; \
# RUN: }" > %t2.script
# RUN: not ld.lld %t --script %t2.script -o %t3 2>&1 | \
# RUN:  FileCheck --check-prefix=ERR1 %s
# ERR1: malformed number: 0x11h

# RUN: echo "SECTIONS { \
# RUN:  . = 0x11k; \
# RUN: }" > %t3.script
# RUN: not ld.lld %t --script %t3.script -o %t4 2>&1 | \
# RUN:  FileCheck --check-prefix=ERR2 %s
# ERR2: malformed number: 0x11k

# RUN: echo "SECTIONS { \
# RUN:  . = 0x11m; \
# RUN: }" > %t4.script
# RUN: not ld.lld %t --script %t4.script -o %t5 2>&1 | \
# RUN:  FileCheck --check-prefix=ERR3 %s
# ERR3: malformed number: 0x11m

.globl _start
_start:
nop

.section .hex.1, "a"
.quad 0

.section .kilo.1, "a"
.quad 0

.section .mega.1, "a"
.quad 0

.section .hex.2, "a"
.quad 0

.section .kilo.2, "a"
.quad 0

.section .mega.2, "a"
.quad 0
