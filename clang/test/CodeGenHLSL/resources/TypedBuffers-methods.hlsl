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

export uint TestGetDimensions() {
    uint dim1, dim2;
    Buf.GetDimensions(dim1);
    RWBuf.GetDimensions(dim2);
    return dim1 + dim2;
}

// CHECK: @TestGetDimensions()()
// CHECK: call void @hlsl::Buffer<float>::GetDimensions(unsigned int&)(ptr {{.*}} @Buf, ptr {{.*}})
// CHECK: call void @hlsl::RWBuffer<unsigned int vector[4]>::GetDimensions(unsigned int&)(ptr {{.*}} @RWBuf, ptr {{.*}})
// CHECK: add
// CHECK: ret

// CHECK: define {{.*}} void @hlsl::Buffer<float>::GetDimensions(unsigned int&)(ptr {{.*}} %this, ptr noalias {{.*}} %dim)
// CHECK: %[[HANDLE_PTR:.*]] = getelementptr inbounds nuw %"class.hlsl::Buffer", ptr %this1, i32 0, i32 0
// CHECK-NEXT: %[[HANDLE:.*]] = load target("dx.TypedBuffer", float, 0, 0, 0), ptr %[[HANDLE_PTR]]
// CHECK-NEXT: %[[DIMPTR:.*]] = load ptr, ptr %dim.addr
// DXIL-NEXT: %[[DIM:.*]] = call i32 @llvm.dx.resource.getdimensions.x.tdx.TypedBuffer_f32_0_0_0t(target("dx.TypedBuffer", float, 0, 0, 0) %[[HANDLE]])
// CHECK-NEXT: store i32 %[[DIM]], ptr %[[DIMPTR]]
// CHECK-NEXT: ret void

// CHECK: define {{.*}} void @hlsl::RWBuffer<unsigned int vector[4]>::GetDimensions(unsigned int&)(ptr {{.*}} %this, {{.*}} %dim)
// CHECK: %[[HANDLE_PTR:.*]] = getelementptr inbounds nuw %"class.hlsl::RWBuffer", ptr %{{.*}}, i32 0, i32 0
// CHECK-NEXT: %[[HANDLE:.*]] = load target("dx.TypedBuffer", <4 x i32>, 1, 0, 0), ptr %[[HANDLE_PTR]]
// CHECK-NEXT: %[[DIMPTR:.*]] = load ptr, ptr %dim.addr
// DXIL-NEXT: %[[DIM:.*]] = call i32 @llvm.dx.resource.getdimensions.x.tdx.TypedBuffer_v4i32_1_0_0t(target("dx.TypedBuffer", <4 x i32>, 1, 0, 0) %[[HANDLE]])
// CHECK-NEXT: store i32 %[[DIM]], ptr %[[DIMPTR]]
// CHECK-NEXT: ret void

// DXIL: declare ptr @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_f32_0_0_0t(target("dx.TypedBuffer", float, 0, 0, 0), i32)
// DXIL: declare ptr @llvm.dx.resource.getpointer.p0.tdx.TypedBuffer_v4i32_1_0_0t(target("dx.TypedBuffer", <4 x i32>, 1, 0, 0), i32)

// DXIL: declare i32 @llvm.dx.resource.getdimensions.x.tdx.TypedBuffer_f32_0_0_0t(target("dx.TypedBuffer", float, 0, 0, 0))
// DXIL: declare i32 @llvm.dx.resource.getdimensions.x.tdx.TypedBuffer_v4i32_1_0_0t(target("dx.TypedBuffer", <4 x i32>, 1, 0, 0))
