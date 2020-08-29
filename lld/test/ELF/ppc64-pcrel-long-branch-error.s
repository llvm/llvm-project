## The test as-is needs a large heap size.
## Disabled until we know how to check for that prerequisite.
# UNSUPPORTED: ppc

# REQUIRES: ppc
# RUN: echo 'SECTIONS { \
# RUN:       .text_low 0x2000: { *(.text_low) } \
# RUN:       .text_high 0x800002000 : { *(.text_high) } \
# RUN:       }' > %t.script

# RUN: llvm-mc -filetype=obj -triple=ppc64le %s -o %t.o
# RUN: not ld.lld -T %t.script %t.o -o /dev/null 2>&1 | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=ppc64le -defsym HIDDEN=1 %s -o %t.o
# RUN: not ld.lld -shared -T %t.script %t.o -o /dev/null 2>&1 | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=ppc64 %s -o %t.o
# RUN: not ld.lld -T %t.script %t.o -o /dev/null 2>&1 | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=ppc64 -defsym HIDDEN=1 %s -o %t.o
# RUN: not ld.lld -shared -T %t.script %t.o -o /dev/null 2>&1 | FileCheck %s

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
