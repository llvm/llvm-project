// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// RUN-DISABLED: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// NOTE: SPIRV codegen for resource methods is not yet implemented

RWStructuredBuffer<float> RWSB1 : register(u0);
RWStructuredBuffer<float> RWSB2 : register(u1);

// CHECK: %"class.hlsl::RWStructuredBuffer" = type { target("dx.RawBuffer", float, 1, 0) }

export void TestIncrementCounter() {
    RWSB1.IncrementCounter();
}

// CHECK: define void @_Z20TestIncrementCounterv()
// CHECK-DXIL: call i32 @llvm.dx.bufferUpdateCounter.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0) %{{[0-9]+}}, i8 1)

export void TestDecrementCounter() {
    RWSB2.DecrementCounter();
}

// CHECK: define void @_Z20TestDecrementCounterv()
// CHECK-DXIL: call i32 @llvm.dx.bufferUpdateCounter.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0) %{{[0-9]+}}, i8 -1)

// CHECK: declare i32 @llvm.dx.bufferUpdateCounter.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0), i8)
