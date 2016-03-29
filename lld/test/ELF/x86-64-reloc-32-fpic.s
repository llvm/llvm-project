# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: not ld.lld -shared %t.o -o %t.so 2>&1 | FileCheck %s
# CHECK: R_X86_64_32 cannot be a dynamic relocation

.long _shared
