// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,DXIL
// RUN-DISABLED: %clang_cc1 -triple spirv-vulkan-library -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,SPIRV

// NOTE: SPIRV codegen for resource methods is not yet implemented

ByteAddressBuffer Buf : register(t0);
RWByteAddressBuffer RWBuf : register(u0);

// DXIL: %"class.hlsl::ByteAddressBuffer" = type { target("dx.RawBuffer", i8, 0, 0) }
// DXIL: %"class.hlsl::RWByteAddressBuffer" = type { target("dx.RawBuffer", i8, 1, 0) }

// DXIL: @Buf = internal global %"class.hlsl::ByteAddressBuffer" poison
// DXIL: @RWBuf = internal global %"class.hlsl::RWByteAddressBuffer" poison

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
