// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,DXIL
// RUN-DISABLED: %clang_cc1 -triple spirv-vulkan-library -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,SPIRV

// NOTE: SPIRV codegen for resource methods is not yet implemented

ByteAddressBuffer Buf : register(t0);
RWByteAddressBuffer RWBuf : register(u0);

// DXIL: %"class.hlsl::ByteAddressBuffer" = type { target("dx.RawBuffer", i8, 0, 0) }
// DXIL: %"class.hlsl::RWByteAddressBuffer" = type { target("dx.RawBuffer", i8, 1, 0) }

// DXIL: @Buf = internal global %"class.hlsl::ByteAddressBuffer" poison
// DXIL: @RWBuf = internal global %"class.hlsl::RWByteAddressBuffer" poison

export float TestLoad() {
    return Buf.Load(0) + RWBuf.Load4(4).w + Buf.Load<float>(20) + RWBuf.Load<float4>(24).w;
}

// CHECK: define {{.*}} float @TestLoad()()
// CHECK: call {{.*}} i32 @hlsl::ByteAddressBuffer::Load(unsigned int)(ptr {{.*}} @Buf, i32 noundef 0)
// CHECK: call {{.*}} <4 x i32> @hlsl::RWByteAddressBuffer::Load4(unsigned int)(ptr {{.*}} @RWBuf, i32 noundef 4)
// CHECK: call {{.*}} float @float hlsl::ByteAddressBuffer::Load<float>(unsigned int)(ptr {{.*}} @Buf, i32 noundef 20)
// CHECK: call {{.*}} <4 x float> @float vector[4] hlsl::RWByteAddressBuffer::Load<float vector[4]>(unsigned int)(ptr {{.*}} @RWBuf, i32 noundef 24)
// CHECK: add
// CHECK: ret float

// CHECK: define {{.*}} i32 @hlsl::ByteAddressBuffer::Load(unsigned int)(ptr {{.*}} %this, i32 noundef %Index)
// CHECK: %__handle = getelementptr inbounds nuw %"class.hlsl::ByteAddressBuffer", ptr %{{.*}}, i32 0, i32 0
// DXIL-NEXT: %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 0, 0), ptr %__handle
// CHECK-NEXT: %[[INDEX:.*]] = load i32, ptr %Index.addr
// DXIL-NEXT: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_0_0t(target("dx.RawBuffer", i8, 0, 0) %[[HANDLE]], i32 %[[INDEX]])
// CHECK-NEXT: %[[VAL:.*]] = load i32, ptr %[[PTR]]
// CHECK-NEXT: ret i32 %[[VAL]]

// CHECK: define {{.*}} <4 x i32> @hlsl::RWByteAddressBuffer::Load4(unsigned int)(ptr {{.*}} %this, i32 noundef %Index)
// CHECK: %__handle = getelementptr inbounds nuw %"class.hlsl::RWByteAddressBuffer", ptr %{{.*}}, i32 0, i32 0
// DXIL-NEXT: %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr %__handle
// CHECK-NEXT: %[[INDEX:.*]] = load i32, ptr %Index.addr
// DXIL-NEXT: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t(target("dx.RawBuffer", i8, 1, 0) %[[HANDLE]], i32 %[[INDEX]])
// CHECK-NEXT: %[[VEC:.*]] = load <4 x i32>, ptr %[[PTR]]
// CHECK-NEXT: ret <4 x i32> %[[VEC]]

// CHECK: define {{.*}} float @float hlsl::ByteAddressBuffer::Load<float>(unsigned int)(ptr {{.*}} %this, i32 noundef %Index)
// CHECK: %__handle = getelementptr inbounds nuw %"class.hlsl::ByteAddressBuffer", ptr %{{.*}}, i32 0, i32 0
// DXIL-NEXT: %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 0, 0), ptr %__handle
// CHECK-NEXT: %[[INDEX:.*]] = load i32, ptr %Index.addr
// DXIL-NEXT: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_0_0t(target("dx.RawBuffer", i8, 0, 0) %[[HANDLE]], i32 %[[INDEX]])
// CHECK-NEXT: %[[VAL:.*]] = load float, ptr %[[PTR]]
// CHECK-NEXT: ret float %[[VAL]]

// CHECK: define {{.*}} <4 x float> @float vector[4] hlsl::RWByteAddressBuffer::Load<float vector[4]>(unsigned int)(ptr {{.*}} %this, i32 noundef %Index)
// CHECK: %__handle = getelementptr inbounds nuw %"class.hlsl::RWByteAddressBuffer", ptr %{{.*}}, i32 0, i32 0
// DXIL-NEXT: %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr %__handle
// CHECK-NEXT: %[[INDEX:.*]] = load i32, ptr %Index.addr
// DXIL-NEXT: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t(target("dx.RawBuffer", i8, 1, 0) %[[HANDLE]], i32 %[[INDEX]])
// CHECK-NEXT: %[[VEC:.*]] = load <4 x float>, ptr %[[PTR]]
// CHECK-NEXT: ret <4 x float> %[[VEC]]

export float TestLoadWithStatus() {
    uint s1, s2, s3, s4;
    float ret = Buf.Load(0, s1) + RWBuf.Load4(4, s2).w + Buf.Load<float>(20, s3) + RWBuf.Load<float4>(24, s4).w;
    ret += float(s1 + s2 + s3 + s4);
    return ret;
}

// CHECK: define {{.*}} float @TestLoadWithStatus()()
// CHECK: call {{.*}} i32 @hlsl::ByteAddressBuffer::Load(unsigned int, unsigned int&)(ptr {{.*}} @Buf, i32 noundef 0, ptr {{.*}} %tmp)
// CHECK: call {{.*}} <4 x i32> @hlsl::RWByteAddressBuffer::Load4(unsigned int, unsigned int&)(ptr {{.*}} @RWBuf, i32 noundef 4, ptr {{.*}} %tmp1)
// CHECK: call {{.*}} float @float hlsl::ByteAddressBuffer::Load<float>(unsigned int, unsigned int&)(ptr {{.*}} @Buf, i32 noundef 20, ptr {{.*}} %tmp4)
// CHECK: call {{.*}} <4 x float> @float vector[4] hlsl::RWByteAddressBuffer::Load<float vector[4]>(unsigned int, unsigned int&)(ptr {{.*}} @RWBuf, i32 noundef 24, ptr {{.*}} %tmp7)
// CHECK: add
// CHECK: ret float

// CHECK: define {{.*}} i32 @hlsl::ByteAddressBuffer::Load(unsigned int, unsigned int&)(ptr {{.*}} %this, i32 noundef %Index, ptr {{.*}} %Status)
// CHECK: %__handle = getelementptr inbounds nuw %"class.hlsl::ByteAddressBuffer", ptr %{{.*}}, i32 0, i32 0
// DXIL-NEXT: %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 0, 0), ptr %__handle
// CHECK-NEXT: %[[INDEX:.*]] = load i32, ptr %Index.addr
// CHECK-NEXT: %[[LOADED_STATUS_ADDR:.*]] = load ptr, ptr %Status.addr
// DXIL-NEXT: %[[STRUCT:.*]] = call { i32, i1 } @llvm.dx.resource.load.rawbuffer.i32.tdx.RawBuffer_i8_0_0t(target("dx.RawBuffer", i8, 0, 0) %[[HANDLE]], i32 %[[INDEX]], i32 poison)
// CHECK-NEXT: %[[VALUE:.*]] = extractvalue { i32, i1 } %[[STRUCT]], 0
// CHECK-NEXT: %[[STATUS_TEMP:.*]] = extractvalue { i32, i1 } %[[STRUCT]], 1
// CHECK-NEXT: %[[STATUS_EXT:.*]] = zext i1 %[[STATUS_TEMP]] to i32
// CHECK-NEXT: store i32 %[[STATUS_EXT]], ptr %[[LOADED_STATUS_ADDR]], align 4
// CHECK-NEXT: ret i32 %[[VALUE]]

// CHECK: define {{.*}} <4 x i32> @hlsl::RWByteAddressBuffer::Load4(unsigned int, unsigned int&)(ptr {{.*}} %this, i32 noundef %Index, ptr {{.*}} %Status)
// CHECK: %__handle = getelementptr inbounds nuw %"class.hlsl::RWByteAddressBuffer", ptr %{{.*}}, i32 0, i32 0
// DXIL-NEXT: %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr %__handle
// CHECK-NEXT: %[[INDEX:.*]] = load i32, ptr %Index.addr
// CHECK-NEXT: %[[LOADED_STATUS_ADDR:.*]] = load ptr, ptr %Status.addr
// DXIL-NEXT: %[[STRUCT:.*]] = call { <4 x i32>, i1 } @llvm.dx.resource.load.rawbuffer.v4i32.tdx.RawBuffer_i8_1_0t(target("dx.RawBuffer", i8, 1, 0) %[[HANDLE]], i32 %[[INDEX]], i32 poison)
// CHECK-NEXT: %[[VALUE:.*]] = extractvalue { <4 x i32>, i1 } %[[STRUCT]], 0
// CHECK-NEXT: %[[STATUS_TEMP:.*]] = extractvalue { <4 x i32>, i1 } %[[STRUCT]], 1
// CHECK-NEXT: %[[STATUS_EXT:.*]] = zext i1 %[[STATUS_TEMP]] to i32
// CHECK-NEXT: store i32 %[[STATUS_EXT]], ptr %[[LOADED_STATUS_ADDR]], align 4
// CHECK-NEXT: ret <4 x i32> %[[VALUE]]

// CHECK: define {{.*}} float @float hlsl::ByteAddressBuffer::Load<float>(unsigned int, unsigned int&)(ptr {{.*}} %this, i32 noundef %Index, ptr {{.*}} %Status)
// CHECK: %__handle = getelementptr inbounds nuw %"class.hlsl::ByteAddressBuffer", ptr %{{.*}}, i32 0, i32 0
// DXIL-NEXT: %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 0, 0), ptr %__handle
// CHECK-NEXT: %[[INDEX:.*]] = load i32, ptr %Index.addr
// CHECK-NEXT: %[[LOADED_STATUS_ADDR:.*]] = load ptr, ptr %Status.addr
// DXIL-NEXT: %[[STRUCT:.*]] = call { float, i1 } @llvm.dx.resource.load.rawbuffer.f32.tdx.RawBuffer_i8_0_0t(target("dx.RawBuffer", i8, 0, 0) %[[HANDLE]], i32 %[[INDEX]], i32 poison)
// CHECK-NEXT: %[[VALUE:.*]] = extractvalue { float, i1 } %[[STRUCT]], 0
// CHECK-NEXT: %[[STATUS_TEMP:.*]] = extractvalue { float, i1 } %[[STRUCT]], 1
// CHECK-NEXT: %[[STATUS_EXT:.*]] = zext i1 %[[STATUS_TEMP]] to i32
// CHECK-NEXT: store i32 %[[STATUS_EXT]], ptr %[[LOADED_STATUS_ADDR]], align 4
// CHECK-NEXT: ret float %[[VALUE]]

// CHECK: define {{.*}} <4 x float> @float vector[4] hlsl::RWByteAddressBuffer::Load<float vector[4]>(unsigned int, unsigned int&)(ptr {{.*}} %this, i32 noundef %Index, ptr {{.*}} %Status)
// CHECK: %__handle = getelementptr inbounds nuw %"class.hlsl::RWByteAddressBuffer", ptr %{{.*}}, i32 0, i32 0
// DXIL-NEXT: %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr %__handle
// CHECK-NEXT: %[[INDEX:.*]] = load i32, ptr %Index.addr
// CHECK-NEXT: %[[LOADED_STATUS_ADDR:.*]] = load ptr, ptr %Status.addr
// DXIL-NEXT: %[[STRUCT:.*]] = call { <4 x float>, i1 } @llvm.dx.resource.load.rawbuffer.v4f32.tdx.RawBuffer_i8_1_0t(target("dx.RawBuffer", i8, 1, 0) %[[HANDLE]], i32 %[[INDEX]], i32 poison)
// CHECK-NEXT: %[[VALUE:.*]] = extractvalue { <4 x float>, i1 } %[[STRUCT]], 0
// CHECK-NEXT: %[[STATUS_TEMP:.*]] = extractvalue { <4 x float>, i1 } %[[STRUCT]], 1
// CHECK-NEXT: %[[STATUS_EXT:.*]] = zext i1 %[[STATUS_TEMP]] to i32
// CHECK-NEXT: store i32 %[[STATUS_EXT]], ptr %[[LOADED_STATUS_ADDR]], align 4
// CHECK-NEXT: ret <4 x float> %[[VALUE]]

export void TestStore() {
    uint4 a;
    float4 b;
    RWBuf.Store(0, a.x);
    RWBuf.Store4(4, a);
    RWBuf.Store<float>(20, b.x);
    RWBuf.Store<float4>(24, b);
    return;
}

// CHECK: define void @TestStore()()
// CHECK: call void @hlsl::RWByteAddressBuffer::Store(unsigned int, unsigned int)(ptr {{.*}} @RWBuf, i32 noundef 0, i32 noundef %{{.*}})
// CHECK: call void @hlsl::RWByteAddressBuffer::Store4(unsigned int, unsigned int vector[4])(ptr {{.*}} @RWBuf, i32 noundef 4, <4 x i32> noundef %{{.*}})
// CHECK: call void @void hlsl::RWByteAddressBuffer::Store<float>(unsigned int, float)(ptr {{.*}} @RWBuf, i32 noundef 20, float noundef {{.*}})
// CHECK: call void @void hlsl::RWByteAddressBuffer::Store<float vector[4]>(unsigned int, float vector[4])(ptr {{.*}} @RWBuf, i32 noundef 24, <4 x float> noundef {{.*}})
// CHECK: ret void

// CHECK: define {{.*}} void @hlsl::RWByteAddressBuffer::Store(unsigned int, unsigned int)(ptr {{.*}} %this, i32 noundef %Index, i32 noundef %Value)
// CHECK: %[[VALUE:.*]] = load i32, ptr %Value.addr
// CHECK-NEXT: %__handle = getelementptr inbounds nuw %"class.hlsl::RWByteAddressBuffer", ptr %{{.*}}, i32 0, i32 0
// DXIL-NEXT: %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr %__handle
// CHECK-NEXT: %[[INDEX:.*]] = load i32, ptr %Index.addr
// DXIL-NEXT: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t(target("dx.RawBuffer", i8, 1, 0) %[[HANDLE]], i32 %[[INDEX]])
// CHECK-NEXT: store i32 %[[VALUE]], ptr %[[PTR]]
// CHECK-NEXT: ret void

// CHECK: define {{.*}} void @hlsl::RWByteAddressBuffer::Store4(unsigned int, unsigned int vector[4])(ptr {{.*}} %this, i32 noundef %Index, <4 x i32> noundef %Value)
// CHECK: %[[VALUE:.*]] = load <4 x i32>, ptr %Value.addr
// CHECK-NEXT: %__handle = getelementptr inbounds nuw %"class.hlsl::RWByteAddressBuffer", ptr %{{.*}}, i32 0, i32 0
// DXIL-NEXT: %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr %__handle
// CHECK-NEXT: %[[INDEX:.*]] = load i32, ptr %Index.addr
// DXIL-NEXT: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t(target("dx.RawBuffer", i8, 1, 0) %[[HANDLE]], i32 %[[INDEX]])
// CHECK-NEXT: store <4 x i32> %[[VALUE]], ptr %[[PTR]]
// CHECK-NEXT: ret void

// CHECK: define {{.*}} void @void hlsl::RWByteAddressBuffer::Store<float>(unsigned int, float)(ptr {{.*}} %this, i32 noundef %Index, float noundef {{.*}} %Value)
// CHECK: %[[VALUE:.*]] = load float, ptr %Value.addr
// CHECK-NEXT: %__handle = getelementptr inbounds nuw %"class.hlsl::RWByteAddressBuffer", ptr %{{.*}}, i32 0, i32 0
// DXIL-NEXT: %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr %__handle
// CHECK-NEXT: %[[INDEX:.*]] = load i32, ptr %Index.addr
// DXIL-NEXT: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t(target("dx.RawBuffer", i8, 1, 0) %[[HANDLE]], i32 %[[INDEX]])
// CHECK-NEXT: store float %[[VALUE]], ptr %[[PTR]]
// CHECK-NEXT: ret void

// CHECK: define {{.*}} void @void hlsl::RWByteAddressBuffer::Store<float vector[4]>(unsigned int, float vector[4])(ptr {{.*}} %this, i32 noundef %Index, <4 x float> noundef {{.*}} %Value)
// CHECK: %[[VALUE:.*]] = load <4 x float>, ptr %Value.addr
// CHECK-NEXT: %__handle = getelementptr inbounds nuw %"class.hlsl::RWByteAddressBuffer", ptr %{{.*}}, i32 0, i32 0
// DXIL-NEXT: %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr %__handle
// CHECK-NEXT: %[[INDEX:.*]] = load i32, ptr %Index.addr
// DXIL-NEXT: %[[PTR:.*]] = call ptr @llvm.dx.resource.getpointer.p0.tdx.RawBuffer_i8_1_0t(target("dx.RawBuffer", i8, 1, 0) %[[HANDLE]], i32 %[[INDEX]])
// CHECK-NEXT: store <4 x float> %[[VALUE]], ptr %[[PTR]]
// CHECK-NEXT: ret void

export uint TestGetDimensions() {
    uint dim1, dim2;
    Buf.GetDimensions(dim1);
    RWBuf.GetDimensions(dim2);
    return dim1 + dim2;
}

// CHECK: define {{.*}} @TestGetDimensions()()
// CHECK: call void @hlsl::ByteAddressBuffer::GetDimensions(unsigned int&)(ptr {{.*}} @Buf, ptr{{.*}})
// CHECK: call void @hlsl::RWByteAddressBuffer::GetDimensions(unsigned int&)(ptr{{.*}} @RWBuf, ptr{{.*}})
// CHECK: add
// CHECK: ret

// CHECK: define {{.*}} void @hlsl::ByteAddressBuffer::GetDimensions(unsigned int&)(ptr {{.*}} %this, {{.*}} %dim)
// CHECK: %[[HANDLE_PTR:.*]] = getelementptr inbounds nuw %"class.hlsl::ByteAddressBuffer", ptr %{{.*}}, i32 0, i32 0
// CHECK-NEXT: %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 0, 0), ptr %[[HANDLE_PTR]]
// CHECK-NEXT: %[[DIMPTR:.*]] = load ptr, ptr %dim.addr
// DXIL-NEXT: %[[DIM:.*]] = call i32 @llvm.dx.resource.getdimensions.x.tdx.RawBuffer_i8_0_0t(target("dx.RawBuffer", i8, 0, 0) %[[HANDLE]])
// CHECK-NEXT: store i32 %[[DIM]], ptr %[[DIMPTR]]
// CHECK-NEXT: ret void

// CHECK: define {{.*}} void @hlsl::RWByteAddressBuffer::GetDimensions(unsigned int&)(ptr {{.*}} %this, ptr noalias {{.*}} %dim)
// CHECK: %[[HANDLE_PTR:.*]] = getelementptr inbounds nuw %"class.hlsl::RWByteAddressBuffer", ptr %{{.*}}, i32 0, i32 0
// CHECK-NEXT: %[[HANDLE:.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr %[[HANDLE_PTR]]
// CHECK-NEXT: %[[DIMPTR:.*]] = load ptr, ptr %dim.addr
// DXIL-NEXT: %[[DIM:.*]] = call i32 @llvm.dx.resource.getdimensions.x.tdx.RawBuffer_i8_1_0t(target("dx.RawBuffer", i8, 1, 0) %[[HANDLE]])
// CHECK-NEXT: store i32 %[[DIM]], ptr %[[DIMPTR]]
// CHECK-NEXT: ret void

// DXIL: declare i32 @llvm.dx.resource.getdimensions.x.tdx.RawBuffer_i8_0_0t(target("dx.RawBuffer", i8, 0, 0))
// DXIL: declare i32 @llvm.dx.resource.getdimensions.x.tdx.RawBuffer_i8_1_0t(target("dx.RawBuffer", i8, 1, 0))
