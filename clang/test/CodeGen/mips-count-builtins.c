// RUN: %clang_cc1 %s -triple mips-unknown-linux-gnu -emit-llvm -o - | FileCheck %s
//
// Test that the ctlz and cttz builtins are defined for zero.
// Based on count-builtin.c

int leading, trailing, pop;

void test_i16(short P) {
  leading = __builtin_clzs(P);
  trailing = __builtin_ctzs(P);

// CHECK: @test_i16
// CHECK: call i16 @llvm.ctlz.i16(i16 {{.*}}, i1 false)
// CHECK: call i16 @llvm.cttz.i16(i16 {{.*}}, i1 false)
}

void test_i32(int P) {
  leading = __builtin_clz(P);
  trailing = __builtin_ctz(P);

// CHECK: @test_i32
// CHECK: call i32 @llvm.ctlz.i32(i32 {{.*}}, i1 false)
// CHECK: call i32 @llvm.cttz.i32(i32 {{.*}}, i1 false)
}

void test_i64(float P) {
  leading = __builtin_clzll(P);
  trailing = __builtin_ctzll(P);
// CHECK: @test_i64
// CHECK: call i64 @llvm.ctlz.i64(i64 {{.*}}, i1 false)
// CHECK: call i64 @llvm.cttz.i64(i64 {{.*}}, i1 false)
}
