// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// RUN: %clang_cc1 -finclude-default-header -triple spirv-vulkan-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// Scenario 1: Basic Padding (No row crossing).
struct Basic {
  float3 a;
  float b;
};
// CHECK-DAG: %Basic = type <{ <3 x float>, float }>
// CHECK-DXIL-DAG: %"class.hlsl::ConstantBuffer" = type { target("dx.CBuffer", %Basic) }
// CHECK-SPIRV-DAG: %"class.hlsl::ConstantBuffer" = type { target("spirv.VulkanBuffer", %Basic, 2, 0) }
ConstantBuffer<Basic> cb_basic;

// Scenario 2: Row Boundary Crossing.
struct RowCrossing {
  float2 a;
  float3 b;
};
// CHECK-DXIL-DAG: %RowCrossing = type <{ <2 x float>, target("dx.Padding", 8), <3 x float> }>
// CHECK-SPIRV-DAG: %RowCrossing = type <{ <2 x float>, target("spirv.Padding", 8), <3 x float> }>
ConstantBuffer<RowCrossing> cb_row_crossing;

// Scenario 3: Arrays.
struct ArrayPadding {
  float a[2];
  float b;
};
// CHECK-DXIL-DAG: %ArrayPadding = type <{ <{ [1 x <{ float, target("dx.Padding", 12) }>], float }>, float }>
// CHECK-SPIRV-DAG: %ArrayPadding = type <{ <{ [1 x <{ float, target("spirv.Padding", 12) }>], float }>, float }>
ConstantBuffer<ArrayPadding> cb_array;

// Scenario 4: Nested Structs.
struct Inner {
  float a;
};
struct Outer {
  Inner i;
  float3 b;
};
// CHECK-DAG: %Inner = type <{ float }>
// CHECK-DAG: %Outer = type <{ %Inner, <3 x float> }>
ConstantBuffer<Outer> cb_nested;

[numthreads(1,1,1)]
void main() {
  // Scenario 1
  // CHECK-LABEL: define {{.*}} void @_Z4mainv()
  // CHECK: %[[CB_BASIC:.*]] = call {{.*}} ptr addrspace({{.*}}) @_ZNK4hlsl14ConstantBufferI5BasicEcvRU{{.*}}S1_Ev
  // CHECK: getelementptr inbounds nuw %Basic, ptr addrspace({{.*}}) %[[CB_BASIC]], i32 0, i32 1
  float f1 = cb_basic.b;

  // Scenario 2
  // CHECK: %[[CB_ROW:.*]] = call {{.*}} ptr addrspace({{.*}}) @_ZNK4hlsl14ConstantBufferI11RowCrossingEcvRU{{.*}}S1_Ev
  // CHECK: getelementptr inbounds nuw %RowCrossing, ptr addrspace({{.*}}) %[[CB_ROW]], i32 0, i32 2
  float3 f2 = cb_row_crossing.b;

  // Scenario 3
  // CHECK: %[[CB_ARRAY:.*]] = call {{.*}} ptr addrspace({{.*}}) @_ZNK4hlsl14ConstantBufferI12ArrayPaddingEcvRU{{.*}}S1_Ev
  // CHECK: getelementptr inbounds nuw %ArrayPadding, ptr addrspace({{.*}}) %[[CB_ARRAY]], i32 0, i32 1
  float f3 = cb_array.b;

  // Scenario 4
  // CHECK: %[[CB_NESTED:.*]] = call {{.*}} ptr addrspace({{.*}}) @_ZNK4hlsl14ConstantBufferI5OuterEcvRU{{.*}}S1_Ev
  // CHECK: getelementptr inbounds nuw %Outer, ptr addrspace({{.*}}) %[[CB_NESTED]], i32 0, i32 1
  float3 f4 = cb_nested.b;
}
