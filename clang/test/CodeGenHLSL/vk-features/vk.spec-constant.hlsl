// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -o - | FileCheck %s


// CHECK-DAG: @_ZL3sc0 = external addrspace(12) constant i32, align 4 [[A0:#[0-9]+]]
// CHECK-DAG: attributes [[A0]] = { "spirv-constant-id"="0,1" }                                                                                                                                                                                                                                                
[[vk::constant_id(0)]]
const bool sc0 = true;

// CHECK-DAG: @_ZL3sc1 = external addrspace(12) constant i32, align 4 [[A1:#[0-9]+]]
// CHECK-DAG: attributes [[A1]] = { "spirv-constant-id"="1,10" }                                                                                                                                                                                                                                               
[[vk::constant_id(1)]]
const int sc1 = 10;

// CHECK-DAG: @_ZL3sc2 = external addrspace(12) constant i32, align 4 [[A2:#[0-9]+]]
// CHECK-DAG: attributes [[A2]] = { "spirv-constant-id"="2,-20" }                                                                                                                                                                                                                                              
[[vk::constant_id(2)]]
const int sc2 = 10-30;

// CHECK-DAG: @_ZL3sc3 = external addrspace(12) constant float, align 4 [[A3:#[0-9]+]]
// CHECK-DAG: attributes [[A3]] = { "spirv-constant-id"="3,0.25" }
[[vk::constant_id(3)]]
const float sc3 = 0.5*0.5;

// CHECK-DAG: @_ZL3sc4 = external addrspace(12) constant i32, align 4 [[A4:#[0-9]+]]
// CHECK-DAG: attributes [[A4]] = { "spirv-constant-id"="4,2" }
enum E {
    A,
    B,
    C
};

[[vk::constant_id(4)]]
const E sc4 = E::C;

[numthreads(1,1,1)]
void main() {
    bool b = sc0;
    int i = sc1;
    int j = sc2;
    float f = sc3;
    E e = sc4;
}
