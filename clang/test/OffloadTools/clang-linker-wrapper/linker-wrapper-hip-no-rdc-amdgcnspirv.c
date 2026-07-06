// REQUIRES: amdgpu-registered-target
// REQUIRES: spirv-registered-target

// In a mixed non-RDC HIP compile, a global --no-lto must drive the concrete
// arch onto the non-LTO pipeline while being ignored for SPIR-V.

// RUN: %clang -cc1 %s -triple amdgcn-amd-amdhsa -emit-llvm-bc -o %t.amdgpu.bc
// RUN: %clang -cc1 %s -triple spirv64-amd-amdhsa -emit-llvm-bc -o %t.spirv.bc
// RUN: llvm-offload-binary -o %t.out \
// RUN:   --image=file=%t.amdgpu.bc,kind=hip,triple=amdgcn-amd-amdhsa,arch=gfx1200 \
// RUN:   --image=file=%t.spirv.bc,kind=hip,triple=spirv64-amd-amdhsa,arch=amdgcnspirv

// RUN: clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --dry-run --no-lto \
// RUN:   --emit-fatbin-only --linker-path=/usr/bin/ld %t.out -o %t.hipfb 2>&1 \
// RUN: | FileCheck %s

// The concrete arch honors --no-lto
// CHECK: clang{{.*}} --target=amdgcn-amd-amdhsa -mcpu=gfx1200 {{.*}}-x ir {{.*}}-flto=none

// SPIR-V ignores the leaked --no-lto
// CHECK: clang{{.*}} --target=spirv64-amd-amdhsa -march=amdgcnspirv
// CHECK-NOT: -x ir
// CHECK-NOT: -flto=none
