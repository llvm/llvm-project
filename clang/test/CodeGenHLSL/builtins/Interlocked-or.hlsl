// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple dxil-pc-shadermodel6.6-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefixes=CHECK

RWByteAddressBuffer buf: register(u0);

// CHECK: %"class.hlsl::RWByteAddressBuffer" = type { target("dx.RawBuffer", i8, 1, 0) }

// CHECK-LABEL: define {{.*}} @_Z11test_return
// CHECK: call void @_ZN4hlsl19RWByteAddressBuffer13InterlockedOrEjjRj
// CHECK: ret i32 {{%.*}}
uint test_return() {
  uint returnVal;
  buf.InterlockedOr(0, 0u, returnVal);
  return returnVal;
}

// CHECK-LABEL: define {{.*}} @_ZN4hlsl19RWByteAddressBuffer13InterlockedOrEjjRj(
// CHECK: [[this_addr:%.*]] = alloca ptr
// CHECK: [[original_val:%.*]] = alloca ptr
// CHECK: [[this:%.*]] = load ptr, ptr [[this_addr]]
// CHECK: [[handle:%.*]] = getelementptr inbounds nuw %"class.hlsl::RWByteAddressBuffer", ptr [[this]], i32 0, i32 0
// CHECK: [[buf:%.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr [[handle]]
// CHECK: [[dest:%.*]] = load i32, ptr %dest.addr
// CHECK: [[newval:%.*]] = load i32, ptr %value.addr
// CHECK: [[result:%.*]] = call i32 @llvm.dx.interlocked.or.i32.tdx.RawBuffer_i8_1_0t.i32(target("dx.RawBuffer", i8, 1, 0) [[buf]], i32 [[dest]], i32 poison, i32 poison, i32 [[newval]])
// CHECK: [[loaded_orig_val_ptr:%.*]] = load ptr, ptr [[original_val]]
// CHECK: store i32 [[result]], ptr [[loaded_orig_val_ptr]]

// CHECK-LABEL: define {{.*}} @_Z12test_returnS
// CHECK: call void @_ZN4hlsl19RWByteAddressBuffer13InterlockedOrEjiRi
// CHECK: ret i32 {{%.*}}
int test_returnS() {
  int returnValS;
  buf.InterlockedOr(0, 0, returnValS);
  return returnValS;
}

// CHECK-LABEL: define {{.*}} @_ZN4hlsl19RWByteAddressBuffer13InterlockedOrEjiRi(
// CHECK: [[this_addr:%.*]] = alloca ptr
// CHECK: [[original_val:%.*]] = alloca ptr
// CHECK: [[this:%.*]] = load ptr, ptr [[this_addr]]
// CHECK: [[handle:%.*]] = getelementptr inbounds nuw %"class.hlsl::RWByteAddressBuffer", ptr [[this]], i32 0, i32 0
// CHECK: [[buf:%.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr [[handle]]
// CHECK: [[dest:%.*]] = load i32, ptr %dest.addr
// CHECK: [[newval:%.*]] = load i32, ptr %value.addr
// CHECK: [[result:%.*]] = call i32 @llvm.dx.interlocked.or.i32.tdx.RawBuffer_i8_1_0t.i32(target("dx.RawBuffer", i8, 1, 0) [[buf]], i32 [[dest]], i32 poison, i32 poison, i32 [[newval]])
// CHECK: [[loaded_orig_val_ptr:%.*]] = load ptr, ptr [[original_val]]
// CHECK: store i32 [[result]], ptr [[loaded_orig_val_ptr]]

// CHECK-LABEL: define {{.*}} @_Z14test_no_return
// CHECK: call void @_ZN4hlsl19RWByteAddressBuffer13InterlockedOrEjj
// CHECK: call void @_ZN4hlsl19RWByteAddressBuffer13InterlockedOrEji
// CHECK: ret void
void test_no_return() {
  buf.InterlockedOr(0, 0u);
  buf.InterlockedOr(0, 0);
}

// CHECK-LABEL: define {{.*}} void @_ZN4hlsl19RWByteAddressBuffer13InterlockedOrEjj(
// CHECK: [[this_addr:%.*]] = alloca ptr
// CHECK: [[this:%.*]] = load ptr, ptr [[this_addr]]
// CHECK: [[handle:%.*]] = getelementptr inbounds nuw %"class.hlsl::RWByteAddressBuffer", ptr [[this]], i32 0, i32 0
// CHECK: [[buf:%.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr [[handle]]
// CHECK: [[dest:%.*]] = load i32, ptr %dest.addr
// CHECK: [[newval:%.*]] = load i32, ptr %value.addr
// CHECK: {{%.*}} = call i32 @llvm.dx.interlocked.or.i32.tdx.RawBuffer_i8_1_0t.i32(target("dx.RawBuffer", i8, 1, 0) [[buf]], i32 [[dest]], i32 poison, i32 poison, i32 [[newval]])
// CHECK: ret void

// CHECK-LABEL: define {{.*}} void @_ZN4hlsl19RWByteAddressBuffer13InterlockedOrEji(
// CHECK: [[this_addr:%.*]] = alloca ptr
// CHECK: [[this:%.*]] = load ptr, ptr [[this_addr]]
// CHECK: [[handle:%.*]] = getelementptr inbounds nuw %"class.hlsl::RWByteAddressBuffer", ptr [[this]], i32 0, i32 0
// CHECK: [[buf:%.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr [[handle]]
// CHECK: [[dest:%.*]] = load i32, ptr %dest.addr
// CHECK: [[newval:%.*]] = load i32, ptr %value.addr
// CHECK: {{%.*}} = call i32 @llvm.dx.interlocked.or.i32.tdx.RawBuffer_i8_1_0t.i32(target("dx.RawBuffer", i8, 1, 0) [[buf]], i32 [[dest]], i32 poison, i32 poison, i32 [[newval]])
// CHECK: ret void

// CHECK-LABEL: define {{.*}} @_Z13test_return64
// CHECK: call void @_ZN4hlsl19RWByteAddressBuffer15InterlockedOr64EjmRm
// CHECK: ret i64 {{%.*}}
uint64_t test_return64() {
  uint64_t returnVal;
  buf.InterlockedOr64(0, 0ul, returnVal);
  return returnVal;
}

// CHECK-LABEL: define {{.*}} void @_ZN4hlsl19RWByteAddressBuffer15InterlockedOr64EjmRm(
// CHECK: [[this_addr:%.*]] = alloca ptr
// CHECK: [[original_val:%.*]] = alloca ptr
// CHECK: [[this:%.*]] = load ptr, ptr [[this_addr]]
// CHECK: [[handle:%.*]] = getelementptr inbounds nuw %"class.hlsl::RWByteAddressBuffer", ptr [[this]], i32 0, i32 0
// CHECK: [[buf:%.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr [[handle]]
// CHECK: [[dest:%.*]] = load i32, ptr %dest.addr
// CHECK: [[newval:%.*]] = load i64, ptr %value.addr
// CHECK: [[result:%.*]] = call i64 @llvm.dx.interlocked.or.i64.tdx.RawBuffer_i8_1_0t.i64(target("dx.RawBuffer", i8, 1, 0) [[buf]], i32 [[dest]], i32 poison, i32 poison, i64 [[newval]])
// CHECK: [[loaded_orig_val_ptr:%.*]] = load ptr, ptr [[original_val]]
// CHECK: store i64 [[result]], ptr [[loaded_orig_val_ptr]]

// CHECK-LABEL: define {{.*}} @_Z14test_return64S
// CHECK: call void @_ZN4hlsl19RWByteAddressBuffer15InterlockedOr64EjlRl
// CHECK: ret i64 {{%.*}}
int64_t test_return64S() {
  int64_t returnValS;
  buf.InterlockedOr64(0, 0l, returnValS);
  return returnValS;
}

// CHECK-LABEL: define {{.*}} void @_ZN4hlsl19RWByteAddressBuffer15InterlockedOr64EjlRl(
// CHECK: [[this_addr:%.*]] = alloca ptr
// CHECK: [[original_val:%.*]] = alloca ptr
// CHECK: [[this:%.*]] = load ptr, ptr [[this_addr]]
// CHECK: [[handle:%.*]] = getelementptr inbounds nuw %"class.hlsl::RWByteAddressBuffer", ptr [[this]], i32 0, i32 0
// CHECK: [[buf:%.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr [[handle]]
// CHECK: [[dest:%.*]] = load i32, ptr %dest.addr
// CHECK: [[newval:%.*]] = load i64, ptr %value.addr
// CHECK: [[result:%.*]] = call i64 @llvm.dx.interlocked.or.i64.tdx.RawBuffer_i8_1_0t.i64(target("dx.RawBuffer", i8, 1, 0) [[buf]], i32 [[dest]], i32 poison, i32 poison, i64 [[newval]])
// CHECK: [[loaded_orig_val_ptr:%.*]] = load ptr, ptr [[original_val]]
// CHECK: store i64 [[result]], ptr [[loaded_orig_val_ptr]]

// CHECK-LABEL: define {{.*}} @_Z16test_no_return64
// CHECK: call void @_ZN4hlsl19RWByteAddressBuffer15InterlockedOr64Ejm
// CHECK: call void @_ZN4hlsl19RWByteAddressBuffer15InterlockedOr64Ejl
// CHECK: ret void
void test_no_return64() {
  buf.InterlockedOr64(0, 0ul);
  buf.InterlockedOr64(0, 0l);
}

// CHECK-LABEL: define {{.*}} void @_ZN4hlsl19RWByteAddressBuffer15InterlockedOr64Ejm(
// CHECK: [[this_addr:%.*]] = alloca ptr
// CHECK: [[this:%.*]] = load ptr, ptr [[this_addr]]
// CHECK: [[handle:%.*]] = getelementptr inbounds nuw %"class.hlsl::RWByteAddressBuffer", ptr [[this]], i32 0, i32 0
// CHECK: [[buf:%.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr [[handle]]
// CHECK: [[dest:%.*]] = load i32, ptr %dest.addr
// CHECK: [[newval:%.*]] = load i64, ptr %value.addr
// CHECK: {{.*}} = call i64 @llvm.dx.interlocked.or.i64.tdx.RawBuffer_i8_1_0t.i64(target("dx.RawBuffer", i8, 1, 0) [[buf]], i32 [[dest]], i32 poison, i32 poison, i64 [[newval]])
// CHECK: ret void

// CHECK-LABEL: define {{.*}} void @_ZN4hlsl19RWByteAddressBuffer15InterlockedOr64Ejl(
// CHECK: [[this_addr:%.*]] = alloca ptr
// CHECK: [[this:%.*]] = load ptr, ptr [[this_addr]]
// CHECK: [[handle:%.*]] = getelementptr inbounds nuw %"class.hlsl::RWByteAddressBuffer", ptr [[this]], i32 0, i32 0
// CHECK: [[buf:%.*]] = load target("dx.RawBuffer", i8, 1, 0), ptr [[handle]]
// CHECK: [[dest:%.*]] = load i32, ptr %dest.addr
// CHECK: [[newval:%.*]] = load i64, ptr %value.addr
// CHECK: {{.*}} = call i64 @llvm.dx.interlocked.or.i64.tdx.RawBuffer_i8_1_0t.i64(target("dx.RawBuffer", i8, 1, 0) [[buf]], i32 [[dest]], i32 poison, i32 poison, i64 [[newval]])
// CHECK: ret void

// CHECK: declare i32 @llvm.dx.interlocked.or.i32.tdx.RawBuffer_i8_1_0t.i32(target("dx.RawBuffer", i8, 1, 0), i32, i32, i32, i32)

// CHECK: declare i64 @llvm.dx.interlocked.or.i64.tdx.RawBuffer_i8_1_0t.i64(target("dx.RawBuffer", i8, 1, 0), i32, i32, i32, i64)
