# REQUIRES: amdgpu
# RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx900 --amdhsa-code-object-version=3 -filetype=obj %s -o %t.o
# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-readobj --file-headers %t.so | FileCheck --check-prefix=COV3 %s

# COV3: OS/ABI: AMDGPU_HSA (0x40)
# COV3: ABIVersion: 1

# RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx900 --amdhsa-code-object-version=4 -filetype=obj %s -o %t.o
# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-readobj --file-headers %t.so | FileCheck --check-prefix=COV4 %s

# COV4: OS/ABI: AMDGPU_HSA (0x40)
# COV4: ABIVersion: 2

# RUN: llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx900 --amdhsa-code-object-version=5 -filetype=obj %s -o %t.o
# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-readobj --file-headers %t.so | FileCheck --check-prefix=COV5 %s

# COV5: OS/ABI: AMDGPU_HSA (0x40)
# COV5: ABIVersion: 3

.text
  s_nop 0x0
  s_endpgm
