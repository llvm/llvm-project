// UNSUPPORTED: system-windows
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// An externally visible variable so static libraries extract.
__attribute__((visibility("protected"), used)) int x;

// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.elf.o
// RUN: %clang -cc1 %s -triple amdgcn-amd-amdhsa -emit-llvm-bc -o %t.amdgpu.bc

// RUN: llvm-offload-binary -o %t.out \
// RUN:   --image=file=%t.amdgpu.bc,kind=openmp,triple=amdgcn-amd-amdhsa,arch=gfx908
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run \
// RUN:   --linker-path=/usr/bin/ld.lld --allow-multiple-definition \
// RUN:   %t.o -o a.out 2>&1 | FileCheck %s

// CHECK: clang{{.*}} -Wl,--allow-multiple-definition
// CHECK: /usr/bin/ld.lld{{.*}} --allow-multiple-definition
