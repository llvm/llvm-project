# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: not ld.lld -shared %t.o -o %t.so 2>&1 | FileCheck %s
# CHECK: can't create dynamic relocation R_X86_64_PC32 against readonly segment

call _shared
