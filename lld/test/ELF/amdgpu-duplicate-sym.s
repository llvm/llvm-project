# REQUIRES: amdgpu
# RUN: llvm-mc -filetype=obj -triple amdgcn-amd-amdhsa -mcpu=gfx1031 --position-independent --relax-relocations %s -o %t.o

# We use lld-link on purpose to exercise -flavor.
# RUN: lld-link -flavor gnu -shared %t.o

        .text
        .amdgcn_target "amdgcn-amd-amdhsa--gfx1031"
        .protected      xxx                     ; @xxx
        .type   xxx,@object
        .data
        .globl  xxx
xxx:
        .long   123                             ; 0x7b

        .addrsig
        .amdgpu_metadata
---
amdhsa.kernels:  []
amdhsa.target:   amdgcn-amd-amdhsa--gfx1031
amdhsa.version:
  - 1
  - 1
...

        .end_amdgpu_metadata
