# REQUIRES: arm
# RUN: llvm-mc -filetype=obj -triple=arm-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o /dev/null 2>&1 | FileCheck %s

# CHECK: {{.*}}.o:(.ARM.attributes): unrecognized format-version: 0x0

.section .ARM.attributes,"a",%0x70000003
  .quad 0
