# RUN: llvm-mc -filetype=obj -triple amdgcn--amdhsa -mcpu=kaveri %s -o %t.o
# RUN: not lld -e kernel0 -flavor gnu %t.o -o %t

# REQUIRES: amdgpu

.hsa_code_object_version 1,0
.hsa_code_object_isa 7,0,0,"AMD","AMDGPU"

.hsatext
.globl kernel0
.align 256
.amdgpu_hsa_kernel kernel0
kernel0:
  s_endpgm
.Lfunc_end0:
  .size kernel0, .Lfunc_end0-kernel0
