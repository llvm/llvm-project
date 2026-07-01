// REQUIRES: amdgpu-registered-target, x86-registered-target

// Check that --device-linker and --device-compiler forward arguments to triples
// considered compatible, not only exact match.

// RUN: %clang -cc1 %s -triple amdgpu9.00-amd-amdhsa -emit-llvm-bc -o %t.amdgpu9.00.bc
// RUN: %clang -cc1 %s -triple amdgpu9-amd-amdhsa -emit-llvm-bc -o %t.amdgpu9.bc
// RUN: %clang -cc1 %s -triple amdgpu10.3-amd-amdhsa -emit-llvm-bc -o %t.amdgpu10.3.bc

// RUN: llvm-offload-binary -o %t.out                                   \
// RUN:   --image=file=%t.amdgpu9.00.bc,kind=openmp,triple=amdgpu9.00-amd-amdhsa,arch=gfx900 \
// RUN:   --image=file=%t.amdgpu9.bc,kind=openmp,triple=amdgpu9-amd-amdhsa \
// RUN:   --image=file=%t.amdgpu10.3.bc,kind=openmp,triple=amdgpu10.3-amd-amdhsa
//
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out
//
// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run --device-compiler=--save-temps \
// RUN:   --device-linker=amdgpu-amd-amdhsa=-ffinite-math-only \
// RUN:   --device-linker=amdgpu9-amd-amdhsa=-fno-signed-zeroes \
// RUN:   --device-linker=amdgpu10.3-amd-amdhsa=-fapprox-func \
// RUN:   --device-compiler=amdgpu-amd-amdhsa=-DALL_TRIPLES \
// RUN:   --device-compiler=amdgpu9-amd-amdhsa=-DAMDGPU9_TRIPLES \
// RUN:   --device-compiler=amdgpu10.3-amd-amdhsa=-DAMDGPU103_TRIPLES \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s

// CHECK: --target=amdgpu9.00-amd-amdhsa -mcpu=gfx900 {{.*}} -Xlinker -ffinite-math-only -Xlinker -fno-signed-zeroes --save-temps -DALL_TRIPLES -DAMDGPU9_TRIPLES
// CHECK: --target=amdgpu9-amd-amdhsa {{.*}} -Xlinker -ffinite-math-only -Xlinker -fno-signed-zeroes --save-temps -DALL_TRIPLES -DAMDGPU9_TRIPLES
// CHECK: --target=amdgpu10.3-amd-amdhsa {{.*}} -Xlinker -ffinite-math-only -Xlinker -fapprox-func --save-temps -DALL_TRIPLES -DAMDGPU103_TRIPLES

__attribute__((visibility("protected"), used)) int x;
