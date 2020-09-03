# REQUIRES: ppc, system-linux
# RUN: echo 'SECTIONS { \
# RUN:       .text_low 0x2000: { *(.text_low) } \
# RUN:       .text_high 0x200002010 : { *(.text_high) } \
# RUN:       }' > %t.script

## In this test, we do not use -o /dev/null like other similar cases do since
## it will fail in some enviroments with out-of-memory errors associated with
## buffering the output in memeory. The test is enabled for ppc linux only since
## writing to an allocated file will cause time out error for this case on freebsd.

# RUN: llvm-mc -filetype=obj -triple=ppc64le %s -o %t.o
# RUN: not ld.lld -T %t.script %t.o -o %t 2>&1 | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=ppc64le -defsym HIDDEN=1 %s -o %t.o
# RUN: not ld.lld -shared -T %t.script %t.o -o %t 2>&1 | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=ppc64 %s -o %t.o
# RUN: not ld.lld -T %t.script %t.o -o %t 2>&1 | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=ppc64 -defsym HIDDEN=1 %s -o %t.o
# RUN: not ld.lld -shared -T %t.script %t.o -o %t 2>&1 | FileCheck %s

# CHECK: error: offset overflow 34 bits, please compile using the large code model

.section .text_low, "ax", %progbits
.globl _start
_start:
  bl high@notoc
  blr

.section .text_high, "ax", %progbits
.ifdef HIDDEN
.hidden high
.endif
.globl high
high:
  blr
