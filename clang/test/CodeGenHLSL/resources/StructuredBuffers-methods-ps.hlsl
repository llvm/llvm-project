// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-pixel -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// RUN-DISABLED: %clang_cc1 -triple spirv-vulkan-pixel -x hlsl -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// NOTE: SPIRV codegen for resource methods is not yet implemented

RWStructuredBuffer<float> RWSB1, RWSB2;
RasterizerOrderedStructuredBuffer<float> ROSB1, ROSB2;

// CHECK: %"class.hlsl::RWStructuredBuffer" = type { target("dx.RawBuffer", float, 1, 0), target("dx.RawBuffer", float, 1, 0) }

export void TestIncrementCounter() {
// CHECK: define void @_Z20TestIncrementCounterv()
// CHECK-DXIL: call i32 @llvm.dx.resource.updatecounter.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0) %{{[0-9]+}}, i8 1)
// CHECK-DXIL: call i32 @llvm.dx.resource.updatecounter.tdx.RawBuffer_f32_1_1t(target("dx.RawBuffer", float, 1, 1) %{{[0-9]+}}, i8 1)
    RWSB1.IncrementCounter();
    ROSB1.IncrementCounter();
}

export void TestDecrementCounter() {
// CHECK: define void @_Z20TestDecrementCounterv()
// CHECK-DXIL: call i32 @llvm.dx.resource.updatecounter.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0) %{{[0-9]+}}, i8 -1)
// CHECK-DXIL: call i32 @llvm.dx.resource.updatecounter.tdx.RawBuffer_f32_1_1t(target("dx.RawBuffer", float, 1, 1) %{{[0-9]+}}, i8 -1)
    RWSB2.DecrementCounter();
    ROSB2.DecrementCounter();
}

export float TestLoad() {
    return ROSB1.Load(10);
}

// CHECK: define noundef nofpclass(nan inf) float @_Z8TestLoadv()
// CHECK: %[[PTR1:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_f32_1_1t(target("dx.RawBuffer", float, 1, 1) %{{[0-9]+}}, i32 %{{[0-9]+}})
// CHECK: %[[VALUE1:.*]] = load float, ptr %[[PTR1]]

// CHECK: declare i32 @llvm.dx.resource.updatecounter.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0), i8)
// CHECK: declare i32 @llvm.dx.resource.updatecounter.tdx.RawBuffer_f32_1_1t(target("dx.RawBuffer", float, 1, 1), i8)
// CHECK: declare ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_f32_1_1t(target("dx.RawBuffer", float, 1, 1), i32)
