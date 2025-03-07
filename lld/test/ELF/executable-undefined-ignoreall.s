# REQUIRES: x86

## In dynamic linking, --unresolved-symbols=ignore-all behaves similar to -shared:
## for PLT relocations to undefined symbols, produce dynamic relocations if we
## emit .dynsym.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %S/Inputs/shared.s -o %ta.o
# RUN: ld.lld -shared -soname=ta %ta.o -o %ta.so
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o %ta.so -o %t --unresolved-symbols=ignore-all -pie
# RUN: llvm-readobj -r %t | FileCheck %s

# CHECK:      Relocations [
# CHECK-NEXT:   Section ({{.*}}) .rela.plt {
# CHECK-NEXT:     R_X86_64_JUMP_SLOT foo 0x0
# CHECK-NEXT:   }
# CHECK-NEXT: ]

_start:
callq foo@PLT
