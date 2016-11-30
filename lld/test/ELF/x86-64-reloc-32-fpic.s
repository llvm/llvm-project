# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: not ld.lld -shared %t.o -o %t.so 2>&1 | FileCheck %s
# CHECK: {{.*}}:(.data+0x0): relocation R_X86_64_32 cannot be used against shared object; recompile with -fPIC.

.data
.long _shared
