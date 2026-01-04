// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// RUN: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SPV

// CHECK-DXIL: %"class.hlsl::RWStructuredBuffer" = type { target("dx.RawBuffer", float, 1, 0), target("dx.RawBuffer", float, 1, 0) }
// CHECK-SPV: %"class.hlsl::RWStructuredBuffer" = type { target("spirv.VulkanBuffer", [0 x float], 12, 1), target("spirv.VulkanBuffer", i32, 12, 1) }

RWStructuredBuffer<float> BufArray[4];

export void foo(int idx) {
  BufArray[0].IncrementCounter();
  BufArray[idx].DecrementCounter();
}

// CHECK: @[[BufArrayStr:.*]] = private unnamed_addr constant [9 x i8] c"BufArray\00", align 1

// CHECK: define {{.*}}void @_Z3fooi(i32 noundef %[[IDX_ARG:.*]])
// CHECK-NEXT: entry:
// CHECK: %[[IDX_ADDR:.*]] = alloca i32
// CHECK: [[TMP_INC:%.*]] = alloca %"class.hlsl::RWStructuredBuffer"
// CHECK: [[TMP_DEC:%.*]] = alloca %"class.hlsl::RWStructuredBuffer"
// CHECK: store i32 %[[IDX_ARG]], ptr %[[IDX_ADDR]]
// CHECK: call void @_ZN4hlsl18RWStructuredBufferIfE46__createFromImplicitBindingWithImplicitCounterEjjijPKcj(ptr {{.*}} [[TMP_INC]], i32 noundef 0, i32 noundef 0, i32 noundef 4, i32 noundef 0, ptr noundef @[[BufArrayStr]], i32 noundef 1)
// CHECK: call noundef i32 @_ZN4hlsl18RWStructuredBufferIfE16IncrementCounterEv(ptr {{.*}} [[TMP_INC]])
// CHECK: %[[IDX_LOADED:.*]] = load i32, ptr %[[IDX_ADDR]]
// CHECK: call void @_ZN4hlsl18RWStructuredBufferIfE46__createFromImplicitBindingWithImplicitCounterEjjijPKcj(ptr {{.*}} [[TMP_DEC]], i32 noundef 0, i32 noundef 0, i32 noundef 4, i32 noundef %[[IDX_LOADED]], ptr noundef @[[BufArrayStr]], i32 noundef 1)
// CHECK: call noundef i32 @_ZN4hlsl18RWStructuredBufferIfE16DecrementCounterEv(ptr {{.*}} [[TMP_DEC]])