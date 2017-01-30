# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump -t %t | FileCheck %s
# CHECK: 0000000000200040 .text 00000000 .hidden __ehdr_start

.text
.global _start, __ehdr_start
_start:
  .quad __ehdr_start

# RUN: ld.lld -r %t.o -o %t.r
# RUN: llvm-objdump -t %t.r | FileCheck %s --check-prefix=RELOCATABLE

# RELOCATABLE: 0000000000000000 *UND* 00000000 __ehdr_start
