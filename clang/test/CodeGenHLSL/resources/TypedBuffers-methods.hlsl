// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,DXIL
// RUN-DISABLED: %clang_cc1 -triple spirv-vulkan-library -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,SPIRV

// NOTE: SPIRV codegen for resource methods is not yet implemented

Buffer<float> Buf : register(t0);
RWBuffer<uint4> RWBuf : register(u0);

// DXIL: %"class.hlsl::Buffer" = type { target("dx.TypedBuffer", float, 0, 0, 0) }
// DXIL: %"class.hlsl::RWBuffer" = type { target("dx.TypedBuffer", <4 x i32>, 1, 0, 0) }

// DXIL: @Buf = internal global %"class.hlsl::Buffer" poison
// DXIL: @RWBuf = internal global %"class.hlsl::RWBuffer" poison

export float TestLoad() {
    return Buf.Load(1) + RWBuf.Load(2).y;
}

// CHECK: define noundef nofpclass(nan inf) float @TestLoad()()
// CHECK: call {{.*}} float @hlsl::Buffer<float>::Load(unsigned int)(ptr {{.*}} @Buf, i32 noundef 1)
// CHECK: call {{.*}} <4 x i32> @hlsl::RWBuffer<unsigned int vector[4]>::Load(unsigned int)(ptr {{.*}} @RWBuf, i32 noundef 2)
// CHECK: add
// CHECK: ret float

// CHECK: define {{.*}} float @hlsl::Buffer<float>::Load(unsigned int)(ptr {{.*}} %this, i32 noundef %Index)
// CHECK: %__handle = getelementptr inbounds nuw %"class.hlsl::Buffer", ptr %{{.*}}, i32 0, i32 0
// DXIL-NEXT: %[[HANDLE:.*]] = load target("dx.TypedBuffer", float, 0, 0, 0), ptr %__handle
// CHECK-NEXT: %[[INDEX:.*]] = load i32, ptr %Index.addr
// DXIL-NEXT: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_f32_0_0_0t(target("dx.TypedBuffer", float, 0, 0, 0) %[[HANDLE]], i32 %[[INDEX]])
// CHECK-NEXT: %[[VAL:.*]] = load float, ptr %[[PTR]]
// CHECK-NEXT: ret float %[[VAL]]

// CHECK: define {{.*}} <4 x i32> @hlsl::RWBuffer<unsigned int vector[4]>::Load(unsigned int)(ptr {{.*}} %this, i32 noundef %Index)
// CHECK: %__handle = getelementptr inbounds nuw %"class.hlsl::RWBuffer", ptr %{{.*}}, i32 0, i32 0
// DXIL-NEXT: %[[HANDLE:.*]] = load target("dx.TypedBuffer", <4 x i32>, 1, 0, 0), ptr %__handle
// CHECK-NEXT: %[[INDEX:.*]] = load i32, ptr %Index.addr
// DXIL-NEXT: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_v4i32_1_0_0t(target("dx.TypedBuffer", <4 x i32>, 1, 0, 0) %[[HANDLE]], i32 %[[INDEX]])
// CHECK-NEXT: %[[VEC:.*]] = load <4 x i32>, ptr %[[PTR]]
// CHECK-NEXT: ret <4 x i32> %[[VEC]]

// DXIL: declare ptr @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_f32_0_0_0t(target("dx.TypedBuffer", float, 0, 0, 0), i32)
// DXIL: declare ptr @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_v4i32_1_0_0t(target("dx.TypedBuffer", <4 x i32>, 1, 0, 0), i32)
