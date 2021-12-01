# REQUIRES: ppc
# RUN: echo 'SECTIONS { \
# RUN:       .text_low 0x2000: { *(.text_low) } \
# RUN:       .text_high 0x2002000 : { *(.text_high) } \
# RUN:       }' > %t.script

# RUN: llvm-mc -filetype=obj -triple=ppc64le %s -o %t.o
# RUN: ld.lld -T %t.script %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn --mcpu=pwr10 %t | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=ppc64le -defsym HIDDEN=1 %s -o %t.o
# RUN: ld.lld -shared -T %t.script %t.o -o %t.so
# RUN: llvm-objdump -d --no-show-raw-insn --mcpu=pwr10 %t.so | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=ppc64 %s -o %t.o
# RUN: ld.lld -T %t.script %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn --mcpu=pwr10 %t | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=ppc64 -defsym HIDDEN=1 %s -o %t.o
# RUN: ld.lld -shared -T %t.script %t.o -o %t.so
# RUN: llvm-objdump -d --no-show-raw-insn --mcpu=pwr10 %t.so | FileCheck %s

# CHECK-LABEL: <_start>:
# CHECK-NEXT:    2000: bl 0x2010
# CHECK-NEXT:          blr
# CHECK-NEXT:          trap
# CHECK-NEXT:          trap

## Callee address - program counter = 0x2002000 - 0x2010 = 33554416
# CHECK-LABEL: <__gep_setup_high>:
# CHECK-NEXT:    2010: paddi 12, 0, 33554416, 1
# CHECK-NEXT:          mtctr 12
# CHECK-NEXT:          bctr

# CHECK-LABEL: <high>:
# CHECK-NEXT:    2002000: blr

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
