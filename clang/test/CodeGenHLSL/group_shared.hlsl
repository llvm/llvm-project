
// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -Wno-error=groupshared-initializer \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan1.3-compute %s -Wno-error=groupshared-initializer \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// Make sure groupshared translated into address space 3.
// CHECK:@a = external hidden addrspace(3) global [10 x float], align 4

 groupshared float a[10];

// CHECK:@b = external hidden addrspace(3) global [10 x float], align 4
 groupshared float b[10] = {1,2,3,4,5,6,7,8,9,10};

 struct S {
   uint4 x;
 };

// CHECK:@c = external hidden addrspace(3) global %struct.S, align 1
extern groupshared S c;

 [numthreads(8,8,1)]
 void main() {
   a[0] = 1;
   b[0] = 1;
   uint d = c.x[0];
 }
