// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-compute -std=hlsl202x -emit-llvm -disable-llvm-passes -hlsl-entry main -o - %s | FileCheck %s

// In the case the template type is specified the groupshared attribute is preserved in the type
// CHECK-LABEL: define linkonce_odr hidden void @_Z4tfooIU3AS3Dv4_iEvT_S2_(ptr addrspace(3) noundef align 16 dereferenceable(16) %a, ptr addrspace(3) noundef align 16 dereferenceable(16) %b)
// CHECK: [[AAddr:%.*]] = alloca ptr addrspace(3), align 4
// CHECK: [[BAddr:%.*]] = alloca ptr addrspace(3), align 4
// CHECK: store ptr addrspace(3) %a, ptr [[AAddr]], align 4
// CHECK: store ptr addrspace(3) %b, ptr [[BAddr]], align 4
// CHECK: [[C:%.*]] = load ptr addrspace(3), ptr [[BAddr]], align 4
// CHECK: [[D:%.*]] = load <4 x i32>, ptr addrspace(3) [[C]], align 16
// CHECK: [[E:%.*]] = load ptr addrspace(3), ptr [[AAddr]], align 4
// CHECK: store <4 x i32> [[D]], ptr addrspace(3) [[E]], align 16
// CHECK: ret void

// In the case the template type is deduced the deduction is done on the non cv-qualified type (the address space is removed)
// So the non groupshared version of the function is deduced
// CHECK-LABEL: define linkonce_odr hidden void @_Z4tfooIDv4_iEvT_S1_(<4 x i32> noundef %a, <4 x i32> noundef %b)
// CHECK: [[AAddr:%.*]] = alloca <4 x i32>, align 16
// CHECK: [[BAddr:%.*]] = alloca <4 x i32>, align 16
// CHECK: store <4 x i32> %a, ptr [[AAddr]], align 16
// CHECK: store <4 x i32> %b, ptr [[BAddr]], align 16
// CHECK: [[C:%.*]] = load <4 x i32>, ptr [[BAddr]], align 16
// CHECK: store <4 x i32> [[C]], ptr [[AAddr]], align 16
// CHECK: ret void
template<typename T>
void tfoo(T a, T b) {
  a = b;
}

using ISF = groupshared int4;
ISF SharedData1;
ISF SharedData2;

[numthreads(4, 1, 1)]
void main(uint3 TID : SV_GroupThreadID) {
  tfoo<ISF>(SharedData1, SharedData2);
  tfoo(SharedData1, SharedData2);
}
