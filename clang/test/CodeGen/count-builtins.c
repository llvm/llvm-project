// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

int leading, trailing, pop;

void test_i16(short P) {
  leading = __builtin_clzs(P);
  trailing = __builtin_ctzs(P);

// CHECK: @test_i16
// CHECK: call i16 @llvm.ctlz.i16
// CHECK: call i16 @llvm.cttz.i16
}

void test_i32(int P) {
  leading = __builtin_clz(P);
  trailing = __builtin_ctz(P);
  pop = __builtin_popcount(P);

// CHECK: @test_i32
// CHECK: call i32 @llvm.ctlz.i32
// CHECK: call i32 @llvm.cttz.i32
// CHECK: call i32 @llvm.ctpop.i32
}

void test_i64(float P) {
  leading = __builtin_clzll(P);
  trailing = __builtin_ctzll(P);
  pop = __builtin_popcountll(P);
// CHECK: @test_i64
// CHECK: call i64 @llvm.ctlz.i64
// CHECK: call i64 @llvm.cttz.i64
// CHECK: call i64 @llvm.ctpop.i64
}
