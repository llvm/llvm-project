// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,DXIL
// RUN-DISABLED: %clang_cc1 -triple spirv-vulkan-library -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,SPV

// NOTE: SPIRV codegen for resource methods is not yet implemented

StructuredBuffer<float> SB1 : register(t0);
RWStructuredBuffer<float> RWSB1 : register(u0);
RWStructuredBuffer<uint4> RWSB2 : register(u1);
AppendStructuredBuffer<float> ASB : register(u2);
ConsumeStructuredBuffer<double> CSB : register(u3);

// DXIL: %"class.hlsl::StructuredBuffer" = type { target("dx.RawBuffer", float, 0, 0) }
// DXIL: %"class.hlsl::RWStructuredBuffer" = type { target("dx.RawBuffer", float, 1, 0), target("dx.RawBuffer", float, 1, 0) }
// DXIL: %"class.hlsl::RWStructuredBuffer.0" = type { target("dx.RawBuffer", <4 x i32>, 1, 0), target("dx.RawBuffer", <4 x i32>, 1, 0) }
// DXIL: %"class.hlsl::AppendStructuredBuffer" = type { target("dx.RawBuffer", float, 1, 0), target("dx.RawBuffer", float, 1, 0) }
// DXIL: %"class.hlsl::ConsumeStructuredBuffer" = type { target("dx.RawBuffer", double, 1, 0), target("dx.RawBuffer", double, 1, 0) }

export int TestIncrementCounter() {
    return RWSB1.IncrementCounter();
}

// CHECK: define noundef i32 @TestIncrementCounter()()
// CHECK: call noundef i32 @hlsl::RWStructuredBuffer<float>::IncrementCounter()(ptr {{.*}} @RWSB1)
// CHECK: ret

// CHECK: define {{.*}} noundef i32 @hlsl::RWStructuredBuffer<float>::IncrementCounter()(ptr {{.*}} %this)
// CHECK: %__handle = getelementptr inbounds nuw %"class.hlsl::RWStructuredBuffer", ptr %{{.*}}, i32 0, i32 0
// DXIL-NEXT: %[[HANDLE:.*]] = load target("dx.RawBuffer", float, 1, 0), ptr %__handle
// DXIL-NEXT: %[[COUNTER:.*]] = call i32 @llvm.dx.resource.updatecounter.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0) %[[HANDLE]], i8 1)
// CHECK-NEXT:  ret i32 %[[COUNTER]]

export int TestDecrementCounter() {
    return RWSB2.DecrementCounter();
}
// CHECK: define {{.*}} i32 @TestDecrementCounter()()
// CHECK: call noundef i32 @hlsl::RWStructuredBuffer<unsigned int vector[4]>::DecrementCounter()(ptr {{.*}} @RWSB2)
// CHECK: ret

// CHECK: define {{.*}} noundef i32 @hlsl::RWStructuredBuffer<unsigned int vector[4]>::DecrementCounter()(ptr {{.*}} %this)
// CHECK: %__handle = getelementptr inbounds nuw %"class.hlsl::RWStructuredBuffer.0", ptr %{{.*}}, i32 0, i32 0
// DXIL-NEXT: %[[HANDLE:.*]] = load target("dx.RawBuffer", <4 x i32>, 1, 0), ptr %__handle
// DXIL-NEXT: %[[COUNTER:.*]] = call i32 @llvm.dx.resource.updatecounter.tdx.RawBuffer_v4i32_1_0t(target("dx.RawBuffer", <4 x i32>, 1, 0) %[[HANDLE]], i8 -1)
// CHECK-NEXT: ret i32 %[[COUNTER]]

export void TestAppend(float value) {
    ASB.Append(value);
}

// CHECK: define void @TestAppend(float)(float {{.*}} %value)
// CHECK: call void @hlsl::AppendStructuredBuffer<float>::Append(float)(ptr {{.*}} @ASB, float noundef nofpclass(nan inf) %0)
// CHECK: ret void

// CHECK: define {{.*}} void @hlsl::AppendStructuredBuffer<float>::Append(float)(ptr {{.*}} %this, float noundef nofpclass(nan inf) %value)
// CHECK: %[[VALUE:.*]] = load float, ptr %value.addr
// CHECK-NEXT: %__handle = getelementptr inbounds nuw %"class.hlsl::AppendStructuredBuffer", ptr %{{.*}}, i32 0, i32 0
// DXIL-NEXT: %[[HANDLE:.*]] = load target("dx.RawBuffer", float, 1, 0), ptr %__handle
// CHECK-NEXT: %__handle2 = getelementptr inbounds nuw %"class.hlsl::AppendStructuredBuffer", ptr %{{.*}}, i32 0, i32 0
// DXIL-NEXT: %[[HANDLE2:.*]] = load target("dx.RawBuffer", float, 1, 0), ptr %__handle2
// DXIL-NEXT: %[[COUNTER:.*]] = call i32 @llvm.dx.resource.updatecounter.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0) %[[HANDLE2]], i8 1)
// DXIL-NEXT: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0) %[[HANDLE]], i32 %[[COUNTER]])
// CHECK-NEXT: store float %[[VALUE]], ptr %[[PTR]]
// CHECK-NEXT: ret void

export double TestConsume() {
    return CSB.Consume();
}
// CHECK: define {{.*}} double @TestConsume()()
// CHECK: call {{.*}} double @hlsl::ConsumeStructuredBuffer<double>::Consume()(ptr {{.*}} @CSB)
// CHECK: ret double
    
// CHECK: define {{.*}} double @hlsl::ConsumeStructuredBuffer<double>::Consume()(ptr {{.*}} %this)
// CHECK: %__handle = getelementptr inbounds nuw %"class.hlsl::ConsumeStructuredBuffer", ptr %{{.*}}, i32 0, i32 0
// DXIL-NEXT: %[[HANDLE:.*]] = load target("dx.RawBuffer", double, 1, 0), ptr %__handle
// CHECK-NEXT: %__handle2 = getelementptr inbounds nuw %"class.hlsl::ConsumeStructuredBuffer", ptr %{{.*}}, i32 0, i32 0
// DXIL-NEXT: %[[HANDLE2:.*]] = load target("dx.RawBuffer", double, 1, 0), ptr %__handle2
// DXIL-NEXT: %[[COUNTER:.*]] = call i32 @llvm.dx.resource.updatecounter.tdx.RawBuffer_f64_1_0t(target("dx.RawBuffer", double, 1, 0) %[[HANDLE2]], i8 -1)
// DXIL-NEXT: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_f64_1_0t(target("dx.RawBuffer", double, 1, 0) %[[HANDLE]], i32 %[[COUNTER]])
// CHECK-NEXT: %[[VAL:.*]] = load double, ptr %[[PTR]], align 8
// CHECK-NEXT: ret double %[[VAL]]

export float TestLoad() {
    return RWSB1.Load(1) + SB1.Load(2);
}

// CHECK: define noundef nofpclass(nan inf) float @TestLoad()()
// CHECK: call {{.*}} float @hlsl::RWStructuredBuffer<float>::Load(unsigned int)(ptr {{.*}} @RWSB1, i32 noundef 1)
// CHECK: call {{.*}} float @hlsl::StructuredBuffer<float>::Load(unsigned int)(ptr {{.*}} @SB1, i32 noundef 2)
// CHECK: add
// CHECK: ret float

// CHECK: define {{.*}} float @hlsl::RWStructuredBuffer<float>::Load(unsigned int)(ptr {{.*}} %this, i32 noundef %Index)
// CHECK: %__handle = getelementptr inbounds nuw %"class.hlsl::RWStructuredBuffer", ptr %{{.*}}, i32 0, i32 0
// DXIL-NEXT: %[[HANDLE:.*]] = load target("dx.RawBuffer", float, 1, 0), ptr %__handle
// CHECK-NEXT: %[[INDEX:.*]] = load i32, ptr %Index.addr
// DXIL-NEXT: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0) %[[HANDLE]], i32 %[[INDEX]])
// CHECK-NEXT: %[[VAL:.*]] = load float, ptr %[[PTR]]
// CHECK-NEXT: ret float %[[VAL]]

// CHECK: define {{.*}} float @hlsl::StructuredBuffer<float>::Load(unsigned int)(ptr {{.*}} %this, i32 noundef %Index)
// CHECK: %__handle = getelementptr inbounds nuw %"class.hlsl::StructuredBuffer", ptr %{{.*}}, i32 0, i32 0
// DXIL-NEXT: %[[HANDLE:.*]] = load target("dx.RawBuffer", float, 0, 0), ptr %__handle
// CHECK-NEXT: %[[INDEX:.*]] = load i32, ptr %Index.addr
// DXIL-NEXT: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_f32_0_0t(target("dx.RawBuffer", float, 0, 0) %[[HANDLE]], i32 %[[INDEX]])
// CHECK-NEXT: %[[VAL:.*]] = load float, ptr %[[PTR]]
// CHECK-NEXT: ret float %[[VAL]]

// DXIL: declare i32 @llvm.dx.resource.updatecounter.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0), i8)
// DXIL: declare i32 @llvm.dx.resource.updatecounter.tdx.RawBuffer_v4i32_1_0t(target("dx.RawBuffer", <4 x i32>, 1, 0), i8)
// DXIL: declare ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_f32_1_0t(target("dx.RawBuffer", float, 1, 0), i32)
// DXIL: declare i32 @llvm.dx.resource.updatecounter.tdx.RawBuffer_f64_1_0t(target("dx.RawBuffer", double, 1, 0), i8)
// DXIL: declare ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_f64_1_0t(target("dx.RawBuffer", double, 1, 0), i32)
// DXIL: declare ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_f32_0_0t(target("dx.RawBuffer", float, 0, 0), i32)
