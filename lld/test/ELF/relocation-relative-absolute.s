# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: not ld.lld %t.o -o %t -pie 2>&1 | FileCheck %s

.globl _start
_start:

# CHECK: relocation R_X86_64_PLT32 cannot refer to absolute symbol answer
call answer@PLT

.globl answer
answer = 42
