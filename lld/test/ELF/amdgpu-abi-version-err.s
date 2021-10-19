# REQUIRES: amdgpu
# RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %s -o %t-0.o
# RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx900 --amdhsa-code-object-version=2 -filetype=obj %s -o %t-1.o
# RUN: not ld.lld -shared %t-0.o %t-1.o -o /dev/null 2>&1 | FileCheck %s

# CHECK: ld.lld: error: incompatible ABI version: {{.*}}-1.o

.text
  s_nop 0x0
  s_endpgm
