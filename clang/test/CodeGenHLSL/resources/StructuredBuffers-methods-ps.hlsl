// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-pixel -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,DXIL
// RUN-DISABLED: %clang_cc1 -triple spirv-vulkan-pixel -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,SPV

// NOTE: SPIRV codegen for resource methods is not yet implemented

RasterizerOrderedStructuredBuffer<float> ROSB1;
RasterizerOrderedStructuredBuffer<int2> ROSB2;

// %"class.hlsl::RasterizerOrderedStructuredBuffer" = type { target("dx.RawBuffer", float, 1, 1), target("dx.RawBuffer", float, 1, 1) }
// %"class.hlsl::RasterizerOrderedStructuredBuffer.0" = type { target("dx.RawBuffer", <2 x i32>, 1, 1), target("dx.RawBuffer", <2 x i32>, 1, 1) }

// CHECK: @ROSB1 = internal global %"class.hlsl::RasterizerOrderedStructuredBuffer" poison
// CHECK: @ROSB2 = internal global %"class.hlsl::RasterizerOrderedStructuredBuffer.0" poison

export void TestIncrementCounter() {
    ROSB1.IncrementCounter();
}

// CHECK: define void @TestIncrementCounter()()
// CHECK: call noundef i32 @hlsl::RasterizerOrderedStructuredBuffer<float>::IncrementCounter()(ptr {{.*}} @ROSB1)
// CHECK-NEXT: ret void

// CHECK: define {{.*}} i32 @hlsl::RasterizerOrderedStructuredBuffer<float>::IncrementCounter()(ptr {{.*}} %this)
// CHECK: %__handle = getelementptr inbounds nuw %"class.hlsl::RasterizerOrderedStructuredBuffer", ptr %{{.*}}, i32 0, i32 0
// CHECK-NEXT: %[[HANDLE:.*]] = load target("dx.RawBuffer", float, 1, 1), ptr %__handle
// DXIL-NEXT: %[[VAL:.*]] = call i32 @llvm.dx.resource.updatecounter.tdx.RawBuffer_f32_1_1t(target("dx.RawBuffer", float, 1, 1) %[[HANDLE]], i8 1)
// CHECK-NEXT: ret i32 %[[VAL]]

export void TestDecrementCounter() {
    ROSB2.DecrementCounter();
}

// CHECK: define void @TestDecrementCounter()()
// CHECK: call noundef i32 @hlsl::RasterizerOrderedStructuredBuffer<int vector[2]>::DecrementCounter()(ptr {{.*}} @ROSB2)
// CHECK-NEXT: ret void

// CHECK: define {{.*}} i32 @hlsl::RasterizerOrderedStructuredBuffer<int vector[2]>::DecrementCounter()(ptr {{.*}} %this)
// CHECK: %__handle = getelementptr inbounds nuw %"class.hlsl::RasterizerOrderedStructuredBuffer.0", ptr %{{.*}}, i32 0, i32 0
// CHECK-NEXT: %[[HANDLE:.*]] = load target("dx.RawBuffer", <2 x i32>, 1, 1), ptr %__handle
// DXIL-NEXT: %[[VAL:.*]] = call i32 @llvm.dx.resource.updatecounter.tdx.RawBuffer_v2i32_1_1t(target("dx.RawBuffer", <2 x i32>, 1, 1) %[[HANDLE]], i8 -1)
// CHECK-NEXT: ret i32 %[[VAL]]

export float TestLoad() {
    return ROSB1.Load(10).x + ROSB2.Load(20).x;
}

// CHECK: define {{.*}} float @TestLoad()()
// CHECK: call {{.*}} float @hlsl::RasterizerOrderedStructuredBuffer<float>::Load(unsigned int)(ptr {{.*}} @ROSB1, i32 noundef 10)
// CHECK: call {{.*}} <2 x i32> @hlsl::RasterizerOrderedStructuredBuffer<int vector[2]>::Load(unsigned int)(ptr {{.*}} @ROSB2, i32 noundef 20)
// CHECK: ret

// CHECK: define {{.*}} float @hlsl::RasterizerOrderedStructuredBuffer<float>::Load(unsigned int)(ptr {{.*}} %Index)
// CHECK: %__handle = getelementptr inbounds nuw %"class.hlsl::RasterizerOrderedStructuredBuffer", ptr {{.*}}, i32 0, i32 0
// CHECK-NEXT: %[[HANDLE:.*]] = load target("dx.RawBuffer", float, 1, 1), ptr %__handle
// CHECK-NEXT: %[[INDEX:.*]] = load i32, ptr %Index.addr
// DXIL-NEXT: %[[BUFPTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_f32_1_1t(target("dx.RawBuffer", float, 1, 1) %[[HANDLE]], i32 %[[INDEX]])
// CHECK-NEXT: %[[VAL:.*]] = load float, ptr %[[BUFPTR]]
// CHECK-NEXT: ret float %[[VAL]]

// CHECK: define {{.*}} <2 x i32> @hlsl::RasterizerOrderedStructuredBuffer<int vector[2]>::Load(unsigned int)(ptr {{.*}} %Index)
// CHECK: %__handle = getelementptr inbounds nuw %"class.hlsl::RasterizerOrderedStructuredBuffer.0", ptr {{.*}}, i32 0, i32 0
// CHECK-NEXT: %[[HANDLE:.*]] = load target("dx.RawBuffer", <2 x i32>, 1, 1), ptr %__handle
// CHECK-NEXT: %[[INDEX:.*]] = load i32, ptr %Index.addr
// DXIL-NEXT: %[[BUFPTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_v2i32_1_1t(target("dx.RawBuffer", <2 x i32>, 1, 1) %[[HANDLE]], i32 %[[INDEX]])
// CHECK-NEXT: %[[VAL:.*]] = load <2 x i32>, ptr %[[BUFPTR]]
// CHECK-NEXT: ret <2 x i32> %[[VAL]]

// DXIL: declare i32 @llvm.dx.resource.updatecounter.tdx.RawBuffer_f32_1_1t(target("dx.RawBuffer", float, 1, 1), i8)
// DXIL: declare i32 @llvm.dx.resource.updatecounter.tdx.RawBuffer_v2i32_1_1t(target("dx.RawBuffer", <2 x i32>, 1, 1), i8)
// DXIL: declare ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_f32_1_1t(target("dx.RawBuffer", float, 1, 1), i32)
// DXIL: declare ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_v2i32_1_1t(target("dx.RawBuffer", <2 x i32>, 1, 1), i32)
