// REQUIRES: amdgpu-registered-target

// Check the readonly feature will can be written to the IR
// if there is no target specified.

// RUN: %clang_cc1 -triple amdgpu -emit-llvm -o - %s | FileCheck --check-prefix=NOCPU %s
// RUN: %clang_cc1 -triple amdgpu9.42 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgpu11.00 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgpu12.00 -emit-llvm -o - %s | FileCheck %s

__attribute__((target("gws,image-insts,vmem-to-lds-load-insts"))) void test() {}

// NOCPU: "target-features"="+gws,+image-insts,+vmem-to-lds-load-insts"
// CHECK-NOT: "target-features"={{.*}}
