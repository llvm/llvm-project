# REQUIRES: amdgpu

# Link new gfx1250 objects with the zero XNACK field used before the subtarget
# feature removal. This must not report an incompatible XNACK mode.
# RUN: split-file %s %t
# RUN: llvm-mc -triple=amdgpu12.50-amd-amdhsa --amdhsa-code-object-version=6 -filetype=obj %t/new.s -o %t/new.o
# RUN: yaml2obj %t/old.yaml -o %t/old.o
# RUN: ld.lld -shared %t/old.o %t/new.o -o %t/linked.so
# RUN: llvm-readobj --file-headers %t/linked.so | FileCheck %s
# RUN: ld.lld -shared %t/new.o %t/old.o -o %t/reversed.so
# RUN: llvm-readobj --file-headers %t/reversed.so | FileCheck %s

# CHECK: Flags [ (0x449)
# CHECK: EF_AMDGPU_FEATURE_SRAMECC_ANY_V4

#--- new.s
s_endpgm

#--- old.yaml
--- !ELF
FileHeader:
  Class: ELFCLASS64
  Data: ELFDATA2LSB
  OSABI: ELFOSABI_AMDGPU_HSA
  ABIVersion: 4
  Type: ET_REL
  Machine: EM_AMDGPU
  Flags: [ EF_AMDGPU_MACH_AMDGCN_GFX1250, EF_AMDGPU_FEATURE_SRAMECC_ANY_V4 ]
