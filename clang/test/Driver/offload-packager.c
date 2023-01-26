// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
// REQUIRES: amdgpu-registered-target
// UNSUPPORTED: system-windows

// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.elf.o

// Check that we can extract files from the packaged binary.
// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_80 \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx908 \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx90a \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx90c 
// RUN: clang-offload-packager %t.out \
// RUN:   --image=file=%t-sm_70.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t-gfx908.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx908
// RUN: diff %t-sm_70.o %t.elf.o
// RUN: diff %t-gfx908.o %t.elf.o

// Check that we generate a new name if one is not given
// RUN: clang-offload-packager -o %t \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_80 \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx908 \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx90a \
// RUN:   --image=file=%t.elf.o,kind=hip,triple=amdgcn-amd-amdhsa,arch=gfx90c 
// RUN: cd $(dirname "%t") && clang-offload-packager %t --image=kind=openmp
// RUN: diff *-nvptx64-nvidia-cuda-sm_70.0.o %t.elf.o; rm *-nvptx64-nvidia-cuda-sm_70.0.o
// RUN: diff *-nvptx64-nvidia-cuda-sm_80.1.o %t.elf.o; rm *-nvptx64-nvidia-cuda-sm_80.1.o
// RUN: diff *-amdgcn-amd-amdhsa-gfx908.2.o %t.elf.o; rm *-amdgcn-amd-amdhsa-gfx908.2.o
// RUN: diff *-amdgcn-amd-amdhsa-gfx90a.3.o %t.elf.o; rm *-amdgcn-amd-amdhsa-gfx90a.3.o
// RUN: not diff *-amdgcn-amd-amdhsa-gfx90c.4.o %t.elf.o

// Check that we can extract from an ELF object file
// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx908 \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-offload-packager %t.out \
// RUN:   --image=file=%t-sm_70.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t-gfx908.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx908
// RUN: diff %t-sm_70.o %t.elf.o
// RUN: diff %t-gfx908.o %t.elf.o

// Check that we can extract from a bitcode file
// RUN: clang-offload-packager -o %t.out \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx908 \
// RUN:   --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-llvm -o %t.bc -fembed-offload-object=%t.out
// RUN: clang-offload-packager %t.out \
// RUN:   --image=file=%t-sm_70.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70 \
// RUN:   --image=file=%t-gfx908.o,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx908
// RUN: diff %t-sm_70.o %t.elf.o
// RUN: diff %t-gfx908.o %t.elf.o
