# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: ld.lld %t -o %t2 --icf=all --print-icf-sections | count 0

.section .foo,"ax"
.byte 42

.section .bar,"ax"
.byte 42
