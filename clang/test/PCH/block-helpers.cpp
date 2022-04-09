// RUN: %clang_cc1 -no-opaque-pointers -x c++-header -triple x86_64-apple-darwin11 -emit-pch -fblocks -fexceptions -o %t %S/block-helpers.h
// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin11 -include-pch %t -emit-llvm -fblocks -fexceptions -o - %s | FileCheck %s

// CHECK: %[[STRUCT_BLOCK_BYREF_X:.*]] = type { i8*, %[[STRUCT_BLOCK_BYREF_X]]*, i32, i32, i8*, i8*, %[[STRUCT_S0:.*]] }
// CHECK: %[[STRUCT_S0]] = type { i32 }
// CHECK: %[[STRUCT_BLOCK_BYREF_Y:.*]] = type { i8*, %[[STRUCT_BLOCK_BYREF_Y]]*, i32, i32, i8*, i8*, %[[STRUCT_S0]] }
// CHECK: %[[STRUCT_BLOCK_DESCRIPTOR:.*]] = type { i64, i64 }

// Check that byref structs are allocated for x and y.

// CHECK-LABEL: define linkonce_odr void @_ZN1S1mEv(
// CHECK: %[[X:.*]] = alloca %[[STRUCT_BLOCK_BYREF_X]], align 8
// CHECK: %[[Y:.*]] = alloca %[[STRUCT_BLOCK_BYREF_Y]], align 8
// CHECK: %[[BLOCK:.*]] = alloca <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8* }>, align 8
// CHECK: %[[BLOCK_CAPTURED:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8* }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8* }>* %[[BLOCK]], i32 0, i32 5
// CHECK: %[[V0:.*]] = bitcast %[[STRUCT_BLOCK_BYREF_X]]* %[[X]] to i8*
// CHECK: store i8* %[[V0]], i8** %[[BLOCK_CAPTURED]], align 8
// CHECK: %[[BLOCK_CAPTURED10:.*]] = getelementptr inbounds <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8* }>, <{ i8*, i32, i32, i8*, %[[STRUCT_BLOCK_DESCRIPTOR]]*, i8*, i8* }>* %[[BLOCK]], i32 0, i32 6
// CHECK: %[[V1:.*]] = bitcast %[[STRUCT_BLOCK_BYREF_Y]]* %[[Y]] to i8*
// CHECK: store i8* %[[V1]], i8** %[[BLOCK_CAPTURED10]], align 8

// CHECK-LABEL: define internal void @___ZN1S1mEv_block_invoke(

// The second call to block_object_assign should be an invoke.

// CHECK-LABEL: define linkonce_odr hidden void @__copy_helper_block_e8_32rc40rc(
// CHECK: call void @_Block_object_assign(
// CHECK: invoke void @_Block_object_assign(
// CHECK: call void @_Block_object_dispose(

// CHECK-LABEL: define linkonce_odr hidden void @__destroy_helper_block_e8_32r40r(
void test() {
  S s;
  s.m();
}
