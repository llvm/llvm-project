# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: ld.lld %t -o %t2
# RUN: llvm-objdump -t -section-headers %t2 | FileCheck %s

## Test __bss_start is defined at the start of .bss

# CHECK: Sections:
# CHECK: Idx Name          Size     VMA                 Type
# CHECK:   2 .bss          00000004 [[ADDR:[0-za-f]+]]  BSS
# CHECK: SYMBOL TABLE:
# CHECK: [[ADDR]]          .bss 00000000 __bss_start

.global __bss_start
.text
_start:
.comm sym1,4,4
