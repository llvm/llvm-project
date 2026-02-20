// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.0-compute -std=hlsl202x -emit-llvm -disable-llvm-passes -hlsl-entry main -o - %s | FileCheck %s

groupshared float4 SharedArr[64];

// CHECK-LABEL: define hidden void @_Z2fnRA64_U3AS3Dv4_ff(ptr addrspace(3) noundef align 16 dereferenceable(1024) %Arr, float noundef nofpclass(nan inf) %F)
// CHECK: [[ArrAddr:%.*]] = alloca ptr addrspace(3), align 4
// CHECK: [[FAddr:%.*]] = alloca float, align 4
// CHECK: store ptr addrspace(3) %Arr, ptr [[ArrAddr]], align 4
// CHECK: store float %F, ptr [[FAddr]], align 4
// CHECK: [[A:%.*]] = load float, ptr [[FAddr]], align 4
// CHECK: [[Splat:%.*]] = insertelement <1 x float> poison, float [[A]], i64 0
// CHECK: [[B:%.*]] = shufflevector <1 x float> [[Splat]], <1 x float> poison, <4 x i32> zeroinitializer
// CHECK: [[C:%.*]] = load ptr addrspace(3), ptr [[ArrAddr]], align 4, !align !3
// CHECK: [[ArrIdx:%.*]] = getelementptr inbounds [64 x <4 x float>], ptr addrspace(3) [[C]], i32 0, i32 5
// CHECK: store <4 x float> [[B]], ptr addrspace(3) [[ArrIdx]], align 16
// CHECK: ret void
void fn(groupshared float4 Arr[64], float F) {
  Arr[5] = F.xxxx;
}

[numthreads(4,1,1)]
void main() {
  fn(SharedArr, 6.0);
}
