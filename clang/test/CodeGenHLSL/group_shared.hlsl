
// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan1.3-compute %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// Make sure groupshared translated into address space 3.
// CHECK:@a = addrspace(3) global [10 x float]

 groupshared float a[10];

 [numthreads(8,8,1)]
 void main() {
   a[0] = 1;
 }

