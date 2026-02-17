// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-compute -std=hlsl202x -emit-llvm -disable-llvm-passes -hlsl-entry main -o - %s | FileCheck %s

groupshared float4 SharedData;

// CHECK-LABEL: define hidden void @_Z3fn1RU3AS3Dv4_f(ptr addrspace(3) noundef align 16 dereferenceable(16) %Sh)
// CHECK: [[ShAddr:%.*]] = alloca ptr addrspace(3), align 4
// CHECK: [[Tmp:%.*]] = alloca <1 x float>, align 4
// CHECK: store ptr addrspace(3) %Sh, ptr [[ShAddr]], align 4
// CHECK: store <1 x float> splat (float 5.000000e+00), ptr [[Tmp]], align 4
// CHECK: [[A:%.*]] = load <1 x float>, ptr [[Tmp]], align 4
// CHECK: [[B:%.*]] = shufflevector <1 x float> [[A]], <1 x float> poison, <4 x i32> zeroinitializer
// CHECK: [[C:%.*]] = load ptr addrspace(3), ptr [[ShAddr]], align 4
// CHECK: store <4 x float> [[B]], ptr addrspace(3) [[C]], align 16
// CHECK: ret void
void fn1(groupshared float4 Sh) {
  Sh = 5.0.xxxx;
}

[numthreads(4, 1, 1)]
void main(uint3 TID : SV_GroupThreadID) {
  fn1(SharedData);
}
