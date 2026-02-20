// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-compute -std=hlsl202x -emit-llvm -disable-llvm-passes -hlsl-entry main -o - %s | FileCheck %s

groupshared float4x4 SharedData;

// CHECK-LABEL: define hidden void @_Z3fn1RU3AS3u11matrix_typeILm4ELm4EfE(ptr addrspace(3) noundef align 4 dereferenceable(64) %Sh)
// CHECK: [[ShAddr:%.*]] = alloca ptr addrspace(3), align 4
// CHECK: store ptr addrspace(3) %Sh, ptr [[ShAddr]], align 4
// CHECK: [[A:%.*]] = load ptr addrspace(3), ptr [[ShAddr]], align 4
// CHECK: [[B:%.*]] = getelementptr <16 x float>, ptr addrspace(3) [[A]], i32 0, i32 4
// CHECK: store float 5.000000e+00, ptr addrspace(3) [[B]], align 4
// CHECK: ret void
void fn1(groupshared float4x4 Sh) {
  Sh[0][1] = 5.0;
}

[numthreads(4,1,1)]
void main(uint3 TID : SV_GroupThreadID) {
  fn1(SharedData);
}
