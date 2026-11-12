// RUN: %clang_cc1 -x c++-header -triple x86_64-apple-darwin11 -emit-pch -O2 -fblocks -fno-escaping-block-tail-calls -o %t %S/no-escaping-block-tail-calls.h
// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -include-pch %t -emit-llvm -O2 -fblocks -fno-escaping-block-tail-calls -o - %s | FileCheck %s

// Check that -fno-escaping-block-tail-calls doesn't disable tail-call
// optimization if the block is non-escaping.

// CHECK-LABEL: define internal noundef i32 @___ZN1S1mEv_block_invoke(
// CHECK: %[[CALL:.*]] = tail call noundef i32 @_ZN1S3fooER2S0(
// CHECK-NEXT: ret i32 %[[CALL]]

void test() {
  S s;
  s.m();
}
