# REQUIRES: x86
## Check that we emit an error for -Ttext-segment.

# RUN: llvm-mc -filetype=obj -triple=x86_64 /dev/null -o %t.o
# RUN: not ld.lld %t.o -Ttext-segment=0x100000 -o /dev/null 2>&1 | FileCheck %s
# RUN: not ld.lld %t.o -Ttext-segment 0x100000 -o /dev/null 2>&1 | FileCheck %s

# CHECK: error: -Ttext-segment is not supported. Use --image-base if you intend to set the base address
