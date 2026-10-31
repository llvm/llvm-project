// Check that the offload wrapper module is created with the correct host
// DataLayout, so the wrapping code emits pointer-size-dependent values using
// the correct pointer width rather than the default layout's.

// REQUIRES: x86-registered-target, amdgpu-registered-target

// RUN: %clang -cc1 -triple i386-unknown-linux-gnu -emit-obj %s -o %t.elf.o
// RUN: %clang -cc1 -triple amdgpu9.00-amd-amdhsa -emit-llvm-bc %s -o %t.amdgpu.bc

// RUN: llvm-offload-binary -o %t.out \
// RUN:   --image=file=%t.amdgpu.bc,kind=openmp,triple=amdgpu9.00-amd-amdhsa
// RUN: %clang -cc1 %s -triple i386-unknown-linux-gnu -emit-obj -o %t.o -fembed-offload-object=%t.out

// RUN: clang-linker-wrapper --host-triple=i386-unknown-linux-gnu --dry-run \
// RUN:   --print-wrapped-module --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 \
// RUN:   | FileCheck %s

// CHECK: target datalayout = "e-m:e-p:32:32-{{.*}}"
// CHECK: target triple = "i386-unknown-linux-gnu"

// The image offsets should use the host's 32-bit size_t type, instead of the
// incorrect i64 according to the default DataLayout.

// CHECK: @.omp_offloading.device_images =
// CHECK-SAME: getelementptr (i8, ptr @.omp_offloading.device_image, i32 {{[0-9]+}})

__attribute__((visibility("protected"), used)) int x;
