// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// RUN-DISABLED: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// NOTE: SPIRV codegen for resource methods is not yet implemented

RWStructuredBuffer<float> RWSB1 : register(u0);
RWStructuredBuffer<float> RWSB2 : register(u1);
AppendStructuredBuffer<float> ASB : register(u2);
ConsumeStructuredBuffer<float> CSB : register(u3);

// CHECK: %"class.hlsl::RWStructuredBuffer" = type { target("dx.RawBuffer", float, 1, 0) }

export int TestIncrementCounter() {
    return RWSB1.IncrementCounter();
}

// CHECK: define noundef i32 @_Z20TestIncrementCounterv()
// CHECK-DXIL: %[[INDEX:.*]] = call i32 @llvm.dx.bufferUpdateCounter.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0) %{{[0-9]+}}, i8 1)
// CHECK-DXIL: ret i32 %[[INDEX]]
export int TestDecrementCounter() {
    return RWSB2.DecrementCounter();
}

// CHECK: define noundef i32 @_Z20TestDecrementCounterv()
// CHECK-DXIL: %[[INDEX:.*]] = call i32 @llvm.dx.bufferUpdateCounter.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0) %{{[0-9]+}}, i8 -1)
// CHECK-DXIL: ret i32 %[[INDEX]]

export void TestAppend(float value) {
    ASB.Append(value);
}

// CHECK: define void @_Z10TestAppendf(float noundef %value)
// CHECK-DXIL: %[[VALUE:.*]] = load float, ptr %value.addr, align 4
// CHECK-DXIL: %[[INDEX:.*]] = call i32 @llvm.dx.bufferUpdateCounter.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0) %{{[0-9]+}}, i8 1)
// CHECK-DXIL: %[[RESPTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0) %{{[0-9]+}}, i32 %[[INDEX]])
// CHECK-DXIL: store float %[[VALUE]], ptr %[[RESPTR]], align 4

export float TestConsume() {
    return CSB.Consume();
}

// CHECK: define noundef float @_Z11TestConsumev()
// CHECK-DXIL: %[[INDEX:.*]] = call i32 @llvm.dx.bufferUpdateCounter.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0) %1, i8 -1)
// CHECK-DXIL: %[[RESPTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0) %0, i32 %[[INDEX]])
// CHECK-DXIL: %[[VALUE:.*]] = load float, ptr %[[RESPTR]], align 4
// CHECK-DXIL: ret float %[[VALUE]]

// CHECK: declare i32 @llvm.dx.bufferUpdateCounter.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0), i8)
// CHECK: declare ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0), i32)
