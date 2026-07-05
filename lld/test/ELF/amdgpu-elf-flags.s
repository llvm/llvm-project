# REQUIRES: amdgpu
# RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx803 --amdhsa-code-object-version=4 -filetype=obj %S/Inputs/amdgpu-kernel-0.s -o %t-0.o
# RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx803 --amdhsa-code-object-version=4 -filetype=obj %S/Inputs/amdgpu-kernel-1.s -o %t-1.o
# RUN: ld.lld -shared %t-0.o %t-1.o -o %t.so
# RUN: llvm-readobj --file-headers %t.so | FileCheck --check-prefix=FIRSTLINK %s

## Try to link again where there are no object file inputs, only a shared library. Issue 47690
# RUN: ld.lld -shared %t.so -o - | llvm-readobj -h - | FileCheck --check-prefix=SECONDLINK %s


# FIRSTLINK:      Flags [
# FIRSTLINK-NEXT:   EF_AMDGPU_MACH_AMDGCN_GFX803 (0x2A)
# FIRSTLINK-NEXT: ]

# SECONDLINK:      Flags [ (0x0)
# SECONDLINK-NEXT: ]
