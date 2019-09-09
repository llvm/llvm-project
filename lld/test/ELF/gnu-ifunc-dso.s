# REQUIRES: x86

## Produce dynamic relocations (symbolic or GOT) for relocations to ifunc
## defined in a DSO.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %S/Inputs/gnu-ifunc-dso.s -o %t1.o
# RUN: ld.lld -shared %t1.o -soname=so -o %t1.so
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld %t.o %t1.so -o %t
# RUN: llvm-readobj --dyn-relocations %t | FileCheck %s
# RUN: ld.lld -shared %t.o %t1.so -o %t.so
# RUN: llvm-readobj --dyn-relocations %t.so | FileCheck %s

# CHECK:      Dynamic Relocations {
# CHECK-NEXT:   R_X86_64_64 bar 0x0
# CHECK-NEXT:   R_X86_64_GLOB_DAT foo 0x0
# CHECK-NEXT: }

.data
  mov foo@gotpcrel(%rip), %rax
 .quad bar
