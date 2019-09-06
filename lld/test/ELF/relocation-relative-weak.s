# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 /dev/null -o %t1.o
# RUN: ld.lld %t1.o -shared -o %t1.so
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld %t.o %t1.so -o %t -pie
# RUN: llvm-readobj --dyn-relocations %t | FileCheck %s

# CHECK:      Dynamic Relocations {
# CHECK-NEXT:   R_X86_64_JUMP_SLOT w 0x0
# CHECK-NEXT: }

.globl _start
_start:

.globl w
.weak w
call w@PLT
