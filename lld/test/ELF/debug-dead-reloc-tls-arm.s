# REQUIRES: arm
## Test we resolve relocations referencing TLS symbols in .debug_* sections to
## a tombstone value if the referenced TLS symbol is discarded.

# RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t.o
# RUN: ld.lld --gc-sections %t.o -o %t
# RUN: llvm-objdump -s %t | FileCheck %s

# CHECK:      Contents of section .debug_info:
# CHECK-NEXT:  0000 ffffffff

.globl _start
_start:
  bx lr

.section .tbss,"awT",%nobits
.globl tls
  .long 0

.section .debug_info
## R_ARM_TLS_LDO32
  .long tls(tlsldo)
