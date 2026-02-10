// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple dxil-pc-shadermodel6.0-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - -DINTERLOCKED32 | \
// RUN:  FileCheck %s --check-prefixes=CHECK-32
// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple dxil-pc-shadermodel6.6-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - -DINTERLOCKED64 | \
// RUN:  FileCheck %s --check-prefixes=CHECK-64

RWByteAddressBuffer buf: register(u0);

// CHECK: %"class.hlsl::RWByteAddressBuffer" = type { target("dx.RawBuffer", i8, 1, 0) }

#ifdef INTERLOCKED32

// CHECK-32-LABEL: define {{.*}} @_Z11test_return
// CHECK-32: call void @_ZN4hlsl19RWByteAddressBuffer13InterlockedOrEjjRj
// CHECK-32: ret i32 {{%.*}}
uint test_return() {
  uint returnVal;
  buf.InterlockedOr(0, 0, returnVal);
  return returnVal;
}

// CHECK-32-LABEL: define {{.*}} @_ZN4hlsl19RWByteAddressBuffer13InterlockedOrEjjRj(
// CHECK-32: [[this_addr:%.*]] = alloca ptr
// CHECK-32: [[original_val:%.*]] = alloca ptr
// CHECK-32: [[this:%.*]] = load ptr, ptr [[this_addr]]
// CHECK-32: [[handle:%.*]] = getelementptr inbounds nuw %"class.hlsl::RWByteAddressBuffer", ptr [[this]], i32 0, i32 0
// CHECK-32: [[buf:%.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr [[handle]]
// CHECK-32: [[dest:%.*]] = load i32, ptr %dest.addr
// CHECK-32: [[newval:%.*]] = load i32, ptr %value.addr
// CHECK-32: [[result:%.*]] = call i32 @llvm.dx.resource.atomicbinop.tdx.RawBuffer_i8_1_0t(target("dx.RawBuffer", i8, 1, 0) [[buf]], i32 2, i32 [[dest]], i32 undef, i32 undef, i32 [[newval]])
// CHECK-32: [[loaded_orig_val_ptr:%.*]] = load ptr, ptr [[original_val]]
// CHECK-32: store i32 [[result]], ptr [[loaded_orig_val_ptr]]

// CHECK-32-LABEL: define {{.*}} @_Z14test_no_return
// CHECK-32: call void @_ZN4hlsl19RWByteAddressBuffer13InterlockedOrEjj
// CHECK-32: ret void
void test_no_return() {
  buf.InterlockedOr(0, 0);
}

// CHECK-32-LABEL: define {{.*}} void @_ZN4hlsl19RWByteAddressBuffer13InterlockedOrEjj(
// CHECK-32: [[this_addr:%.*]] = alloca ptr
// CHECK-32: [[this:%.*]] = load ptr, ptr [[this_addr]]
// CHECK-32: [[handle:%.*]] = getelementptr inbounds nuw %"class.hlsl::RWByteAddressBuffer", ptr [[this]], i32 0, i32 0
// CHECK-32: [[buf:%.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr [[handle]]
// CHECK-32: [[dest:%.*]] = load i32, ptr %dest.addr
// CHECK-32: [[newval:%.*]] = load i32, ptr %value.addr
// CHECK-32: {{%.*}} = call i32 @llvm.dx.resource.atomicbinop.tdx.RawBuffer_i8_1_0t(target("dx.RawBuffer", i8, 1, 0) [[buf]], i32 2, i32 [[dest]], i32 undef, i32 undef, i32 [[newval]])
// CHECK-32: ret void

// CHECK-32: declare i32 @llvm.dx.resource.atomicbinop.tdx.RawBuffer_i8_1_0t(target("dx.RawBuffer", i8, 1, 0), i32, i32, i32, i32, i32)

#endif

#ifdef INTERLOCKED64

// CHECK-LABEL: define {{.*}} @_Z13test_return64
// CHECK: call void @_ZN4hlsl19RWByteAddressBuffer15InterlockedOr64EjmRm
// CHECK: ret i64 {{%.*}}
uint64_t test_return64() {
  uint64_t returnVal;
  buf.InterlockedOr64(0, 0, returnVal);
  return returnVal;
}

// CHECK-64-LABEL: define {{.*}} void @_ZN4hlsl19RWByteAddressBuffer15InterlockedOr64EjmRm(
// CHECK-64: [[this_addr:%.*]] = alloca ptr
// CHECK-64: [[original_val:%.*]] = alloca ptr
// CHECK-64: [[this:%.*]] = load ptr, ptr [[this_addr]]
// CHECK-64: [[handle:%.*]] = getelementptr inbounds nuw %"class.hlsl::RWByteAddressBuffer", ptr [[this]], i32 0, i32 0
// CHECK-64: [[buf:%.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr [[handle]]
// CHECK-64: [[dest:%.*]] = load i32, ptr %dest.addr
// CHECK-64: [[newval:%.*]] = load i64, ptr %value.addr
// CHECK-64: [[result:%.*]] = call i64 @llvm.dx.resource.atomicbinop64.tdx.RawBuffer_i8_1_0t(target("dx.RawBuffer", i8, 1, 0) [[buf]], i32 2, i32 [[dest]], i32 undef, i32 undef, i64 [[newval]])
// CHECK-64: [[loaded_orig_val_ptr:%.*]] = load ptr, ptr [[original_val]]
// CHECK-64: store i64 [[result]], ptr [[loaded_orig_val_ptr]]

// CHECK-LABEL: define {{.*}} @_Z16test_no_return64
// CHECK: call void @_ZN4hlsl19RWByteAddressBuffer15InterlockedOr64Ejm
// CHECK: ret void
void test_no_return64() {
  buf.InterlockedOr64(0, 0);
}

// CHECK-64-LABEL: define {{.*}} void @_ZN4hlsl19RWByteAddressBuffer15InterlockedOr64Ejm(
// CHECK-64: [[this_addr:%.*]] = alloca ptr
// CHECK-64: [[this:%.*]] = load ptr, ptr [[this_addr]]
// CHECK-64: [[handle:%.*]] = getelementptr inbounds nuw %"class.hlsl::RWByteAddressBuffer", ptr [[this]], i32 0, i32 0
// CHECK-64: [[buf:%.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr [[handle]]
// CHECK-64: [[dest:%.*]] = load i32, ptr %dest.addr
// CHECK-64: [[newval:%.*]] = load i64, ptr %value.addr
// CHECK-64: {{.*}} = call i64 @llvm.dx.resource.atomicbinop64.tdx.RawBuffer_i8_1_0t(target("dx.RawBuffer", i8, 1, 0) [[buf]], i32 2, i32 [[dest]], i32 undef, i32 undef, i64 [[newval]])
// CHECK-64: ret void

// CHECK-64: declare i64 @llvm.dx.resource.atomicbinop64.tdx.RawBuffer_i8_1_0t(target("dx.RawBuffer", i8, 1, 0), i32, i32, i32, i32, i64)

#endif
