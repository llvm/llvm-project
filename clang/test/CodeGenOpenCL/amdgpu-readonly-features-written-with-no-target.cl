// REQUIRES: amdgpu-registered-target

// Check the readonly feature will can be written to the IR
// if there is no target specified.

// RUN: %clang_cc1 -triple amdgcn -emit-llvm -o - %s | FileCheck --check-prefix=NOCPU %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx942 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1100 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn -target-cpu gfx1200 -emit-llvm -o - %s | FileCheck %s

__attribute__((target("gws,image-insts,vmem-to-lds-load-insts"))) void test() {}

// NOCPU: "target-features"="+gws,+image-insts,+vmem-to-lds-load-insts"
// CHECK-NOT: "target-features"={{.*}}
