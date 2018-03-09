# REQUIRES: arm
# RUN: llvm-mc -filetype=obj -triple=armv7-unknown-linux-gnueabi %s -o %t.o
# RUN: echo "SECTIONS { .trap : { *(.ARM.exidx) *(.dummy) } }" > %t.script

## We incorrectly removed unused synthetic sections and crashed before.
## Check we do not crash and do not produce .trap output section.
# RUN: ld.lld -shared -o %t.so --script %t.script %t.o
# RUN: llvm-objdump -section-headers %t.so | FileCheck %s
# CHECK-NOT: .trap
