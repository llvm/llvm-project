// RUN: %clang_cc1 -finclude-default-header -emit-llvm -disable-llvm-passes  \
// RUN:   -triple dxil-pc-shadermodel6.3-library %s -o - | FileCheck %s

// Test groupshared vector element store for uint.
// CHECK-LABEL: test_uint4
// CHECK: [[VAL:%.*]] = load i32, ptr %Val.addr, align 4
// CHECK: [[IDX:%.*]] = load i32, ptr %Idx.addr, align 4
// CHECK: [[PTR:%.*]] = getelementptr <4 x i32>, ptr addrspace(3) @SMem, i32 0, i32 [[IDX]]
// CHECK: store i32 [[VAL]], ptr addrspace(3) [[PTR]], align 4
// CHECK-: ret void
groupshared uint4 SMem;
void test_uint4(uint Idx, uint Val) {
  SMem[Idx] = Val;
}

// Test local vector element store for bool.
// CHECK: [[COND1:%.*]] = load i32, ptr addrspace(3) @Cond, align 4
// CHECK: [[COND2:%.*]] = trunc i32 [[COND1]] to i1
// CHECK: [[IDX:%.*]] = load i32, ptr %Idx.addr, align 4
// CHECK: [[COND3:%.*]] = zext i1 [[COND2]] to i32
// CHECK: [[PTR:%.*]] = getelementptr <3 x i32>, ptr %Val, i32 0, i32 [[IDX]]
// CHECK: store i32 [[COND3]], ptr [[PTR]], align 4
// CHECK: ret
groupshared bool Cond;
bool3 test_bool(uint Idx) {
  bool3 Val = { false, false, false};
  Val[Idx] = Cond;
  return Val;
}

// Test resource vector element store for float.
// CHECK: [[VAL:%.*]] = load float, ptr %Val.addr, align 4
// CHECK: [[RES_PTR:%.*]] = call {{.*}} ptr @_ZN4hlsl18RWStructuredBufferIDv4_fEixEj(ptr {{.*}} @_ZL3Buf, i32 noundef 0)
// CHECK: [[IDX:%.*]] = load i32, ptr %Idx.addr, align 4
// CHECK: [[PTR:%.*]] = getelementptr <4 x float>, ptr [[RES_PTR]], i32 0, i32 [[IDX]]
// CHECK: store float [[VAL]], ptr [[PTR]], align 4
// CHECK: ret void
RWStructuredBuffer<float4> Buf : register(u0);
void test_float(uint Idx, float Val) {
  Buf[0][Idx] = Val;
}
