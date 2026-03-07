// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple dxil-pc-shadermodel6.6-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - -DBYTEADDRESS | FileCheck %s --check-prefixes=CHECK-BYTEADDRESS
// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple dxil-pc-shadermodel6.6-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - -DINTBUF | FileCheck %s --check-prefixes=CHECK-INTBUF
// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple dxil-pc-shadermodel6.6-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - -DUINTBUF | FileCheck %s --check-prefixes=CHECK-UINTBUF
// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple dxil-pc-shadermodel6.6-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - -DLONGBUF | FileCheck %s --check-prefixes=CHECK-LONGBUF
// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple dxil-pc-shadermodel6.6-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - -DULONGBUF | FileCheck %s --check-prefixes=CHECK-ULONGBUF
// RUN: %clang_cc1 -finclude-default-header  -x hlsl  -triple dxil-pc-shadermodel6.6-library %s \
// RUN:  -emit-llvm -disable-llvm-passes -o - -DSTRUCTURED | FileCheck %s --check-prefixes=CHECK-STRUCTURED

#ifdef BYTEADDRESS
using handle_t = __hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::raw_buffer]] [[hlsl::contained_type(char)]];
using ret_ty = unsigned int;
#define INVAL 0u
#endif
#ifdef INTBUF
using handle_t = __hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::contained_type(int)]];
using ret_ty = int;
#define INVAL 0
#endif
#ifdef UINTBUF
using handle_t = __hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::contained_type(unsigned int)]];
using ret_ty = unsigned int;
#define INVAL 0u
#endif
#ifdef LONGBUF
using handle_t = __hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::contained_type(long)]];
using ret_ty = long;
#define INVAL 0l
#endif
#ifdef ULONGBUF
using handle_t = __hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::contained_type(unsigned long)]];
using ret_ty = unsigned long;
#define INVAL 0ul
#endif
#ifdef STRUCTURED
struct TestStruct {
  int a;
  unsigned int b;
};

using handle_t = __hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::raw_buffer]] [[hlsl::contained_type(TestStruct)]];
#endif

struct CustomResource {
  handle_t h;
};

#ifndef STRUCTURED

// CHECK-LABEL: define {{.*}} i32 @_Z11test_return14CustomResource(
// CHECK-BYTEADDRESS: %hlsl.interlocked.or = call i32 @llvm.dx.interlocked.or.i32.tdx.RawBuffer_i8_1_0t.i32(target("dx.RawBuffer", i8, 1, 0) {{%.*}}, i32 1, i32 poison, i32 poison, i32 0)
// CHECK-INTBUF: %hlsl.interlocked.or = call i32 @llvm.dx.interlocked.or.i32.tdx.TypedBuffer_i32_1_0_1t.i32(target("dx.TypedBuffer", i32, 1, 0, 1) {{%.*}}, i32 1, i32 poison, i32 poison, i32 0)
// CHECK-UINTBUF: %hlsl.interlocked.or = call i32 @llvm.dx.interlocked.or.i32.tdx.TypedBuffer_i32_1_0_0t.i32(target("dx.TypedBuffer", i32, 1, 0, 0) {{%.*}}, i32 1, i32 poison, i32 poison, i32 0)
// CHECK-LONGBUF: %hlsl.interlocked.or = call i64 @llvm.dx.interlocked.or.i64.tdx.TypedBuffer_i64_1_0_1t.i64(target("dx.TypedBuffer", i64, 1, 0, 1) {{%.*}}, i32 1, i32 poison, i32 poison, i64 0)
// CHECK-ULONGBUF: %hlsl.interlocked.or = call i64 @llvm.dx.interlocked.or.i64.tdx.TypedBuffer_i64_1_0_0t.i64(target("dx.TypedBuffer", i64, 1, 0, 0) {{%.*}}, i32 1, i32 poison, i32 poison, i64 0)
// CHECK-NEXT: store i32 %hlsl.interlocked.or, ptr [[returnVal:%.*]], align 4
// CHECK-NEXT: [[loadedReturnVal:%.*]] = load i32, ptr [[returnVal]], align 4
// CHECK-NEXT: ret i32 [[loadedReturnVal]]
ret_ty test_return(CustomResource cr) {
  ret_ty returnVal = 0;
  __builtin_hlsl_interlocked_or_ret_uint(cr.h, 1u, INVAL, returnVal);
  return returnVal;
}

// CHECK-LABEL: define {{.*}} void @_Z14test_no_return14CustomResource(
// CHECK-BYTEADDRESS: %hlsl.interlocked.or = call i32 @llvm.dx.interlocked.or.i32.tdx.RawBuffer_i8_1_0t.i32(target("dx.RawBuffer", i8, 1, 0) {{%.*}}, i32 1, i32 poison, i32 poison, i32 0)
// CHECK-INTBUF: %hlsl.interlocked.or = call i32 @llvm.dx.interlocked.or.i32.tdx.TypedBuffer_i32_1_0_1t.i32(target("dx.TypedBuffer", i32, 1, 0, 1) {{%.*}}, i32 1, i32 poison, i32 poison, i32 0)
// CHECK-UINTBUF: %hlsl.interlocked.or = call i32 @llvm.dx.interlocked.or.i32.tdx.TypedBuffer_i32_1_0_0t.i32(target("dx.TypedBuffer", i32, 1, 0, 0) {{%.*}}, i32 1, i32 poison, i32 poison, i32 0)
// CHECK-LONGBUF: %hlsl.interlocked.or = call i64 @llvm.dx.interlocked.or.i64.tdx.TypedBuffer_i64_1_0_1t.i64(target("dx.TypedBuffer", i64, 1, 0, 1) {{%.*}}, i32 1, i32 poison, i32 poison, i64 0)
// CHECK-ULONGBUF: %hlsl.interlocked.or = call i64 @llvm.dx.interlocked.or.i64.tdx.TypedBuffer_i64_1_0_0t.i64(target("dx.TypedBuffer", i64, 1, 0, 0) {{%.*}}, i32 1, i32 poison, i32 poison, i64 0)
// CHECK-NEXT: ret void
void test_no_return(CustomResource h) {
  __builtin_hlsl_interlocked_or(h.h, 1u, INVAL);
}

#else

// CHECK-STRUCTURED-LABEL: define {{.*}} i32 @_Z11test_return14CustomResource(
// CHECK-STRUCTURED: %hlsl.interlocked.or = call i32 @llvm.dx.interlocked.or.i32.tdx.RawBuffer_s_struct.TestStructs_1_0t.i32(target("dx.RawBuffer", %struct.TestStruct, 1, 0) %0, i32 1, i32 4, i32 poison, i32 0)
// CHECK-STRUCTURED-NEXT: store i32 %hlsl.interlocked.or, ptr [[returnVal:%.*]], align 4
// CHECK-STRUCTURED-NEXT: [[loadedReturnVal:%.*]] = load i32, ptr [[returnVal]], align 4
// CHECK-STRUCTURED-NEXT: ret i32 [[loadedReturnVal]]
unsigned int test_return(CustomResource cr) {
  unsigned int returnVal = 0u;
  __builtin_hlsl_interlocked_or_ret_uint(cr.h, 1u, 4u, 0u, returnVal);
  return returnVal;
}

// CHECK-STRUCTURED-LABEL: define {{.*}} void @_Z14test_no_return14CustomResource(
// CHECK-STRUCTURED: %hlsl.interlocked.or = call i32 @llvm.dx.interlocked.or.i32.tdx.RawBuffer_s_struct.TestStructs_1_0t.i32(target("dx.RawBuffer", %struct.TestStruct, 1, 0) %0, i32 1, i32 4, i32 poison, i32 0)
// CHECK-STRUCTURED-NEXT: ret void
void test_no_return(CustomResource h) {
  __builtin_hlsl_interlocked_or(h.h, 1u, 4u, 0u);
}

#endif
