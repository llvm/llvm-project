// RUN: %clang_cc1 -x c++-header -triple x86_64-apple-darwin11 -emit-pch -fblocks -fexceptions -o %t %S/block-helpers.h
// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -include-pch %t -emit-llvm -fblocks -fexceptions -o - %s | FileCheck %s

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
