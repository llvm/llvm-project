# RUN: llvm-mc -filetype=obj -triple=amdgcn--amdhsa -mcpu=fiji %s -o %t.o
# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-readobj -r %t.so | FileCheck %s

# REQUIRES: amdgpu

# Make sure that the reloc for local_var is resolved by lld.

  .text

kernel0:
  s_mov_b32 s0, local_var+4
  s_endpgm

  .local local_var

# CHECK: Relocations [
# CHECK-NEXT: ]
