// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// RUN-DISABLED: %clang_cc1 -triple spirv-vulkan-library -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// NOTE: SPIRV codegen for resource methods is not yet implemented

StructuredBuffer<float> SB1 : register(t0);
RWStructuredBuffer<float> RWSB1 : register(u0);
RWStructuredBuffer<float> RWSB2 : register(u1);
AppendStructuredBuffer<float> ASB : register(u2);
ConsumeStructuredBuffer<float> CSB : register(u3);

// CHECK: %"class.hlsl::StructuredBuffer" = type { target("dx.RawBuffer", float, 0, 0) }
// CHECK: %"class.hlsl::RWStructuredBuffer" = type { target("dx.RawBuffer", float, 1, 0), target("dx.RawBuffer", float, 1, 0) }
// CHECK: %"class.hlsl::AppendStructuredBuffer" = type { target("dx.RawBuffer", float, 1, 0), target("dx.RawBuffer", float, 1, 0) }
// CHECK: %"class.hlsl::ConsumeStructuredBuffer" = type { target("dx.RawBuffer", float, 1, 0), target("dx.RawBuffer", float, 1, 0) }

export int TestIncrementCounter() {
    return RWSB1.IncrementCounter();
}

// CHECK: define noundef i32 @_Z20TestIncrementCounterv()
// CHECK-DXIL: %[[INDEX:.*]] = call i32 @llvm.dx.resource.updatecounter.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0) %{{[0-9]+}}, i8 1)
// CHECK-DXIL: ret i32 %[[INDEX]]
export int TestDecrementCounter() {
    return RWSB2.DecrementCounter();
}

// CHECK: define noundef i32 @_Z20TestDecrementCounterv()
// CHECK-DXIL: %[[INDEX:.*]] = call i32 @llvm.dx.resource.updatecounter.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0) %{{[0-9]+}}, i8 -1)
// CHECK-DXIL: ret i32 %[[INDEX]]

export void TestAppend(float value) {
    ASB.Append(value);
}

// CHECK: define void @_Z10TestAppendf(float noundef nofpclass(nan inf) %value)
// CHECK-DXIL: %[[VALUE:.*]] = load float, ptr %value.addr, align 4
// CHECK-DXIL: %[[INDEX:.*]] = call i32 @llvm.dx.resource.updatecounter.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0) %{{[0-9]+}}, i8 1)
// CHECK-DXIL: %[[RESPTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0) %{{[0-9]+}}, i32 %[[INDEX]])
// CHECK-DXIL: store float %[[VALUE]], ptr %[[RESPTR]], align 4

export float TestConsume() {
    return CSB.Consume();
}

// CHECK: define noundef nofpclass(nan inf) float @_Z11TestConsumev()
// CHECK-DXIL: %[[INDEX:.*]] = call i32 @llvm.dx.resource.updatecounter.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0) %1, i8 -1)
// CHECK-DXIL: %[[RESPTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0) %0, i32 %[[INDEX]])
// CHECK-DXIL: %[[VALUE:.*]] = load float, ptr %[[RESPTR]], align 4
// CHECK-DXIL: ret float %[[VALUE]]

export float TestLoad() {
    return RWSB1.Load(1) + SB1.Load(2);
}

// CHECK: define noundef nofpclass(nan inf) float @_Z8TestLoadv()
// CHECK: %[[PTR1:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0) %{{[0-9]+}}, i32 %{{[0-9]+}})
// CHECK: %[[VALUE1:.*]] = load float, ptr %[[PTR1]]
// CHECK: %[[PTR2:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_f32_0_0t(target("dx.RawBuffer", float, 0, 0) %{{[0-9]+}}, i32 %{{[0-9]+}})
// CHECK: %[[VALUE2:.*]] = load float, ptr %[[PTR2]]

// CHECK: declare i32 @llvm.dx.resource.updatecounter.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0), i8)
// CHECK: declare ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0), i32)
// CHECK: declare ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_f32_0_0t(target("dx.RawBuffer", float, 0, 0), i32)
