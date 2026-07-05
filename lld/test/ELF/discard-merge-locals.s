# REQUIRES: x86

## Test that the .L symbol in a SHF_MERGE section is omitted.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-readelf -s %t | FileCheck /dev/null --implicit-check-not=.L.str

lea .L.str(%rip), %rdi
lea local(%rip), %rdi

.section .rodata.str1.1,"aMS",@progbits,1
.L.str:
local:
