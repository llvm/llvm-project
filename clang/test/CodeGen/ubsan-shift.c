// RUN: %clang_cc1 -triple=x86_64-apple-darwin -fsanitize=shift-exponent,shift-base -emit-llvm %s -o - | FileCheck %s

// CHECK-LABEL: define i32 @f1
int f1(int c, int shamt) {
// CHECK: icmp ule i32 %{{.*}}, 31, !nosanitize
// CHECK: icmp ule i32 %{{.*}}, 31, !nosanitize
  return 1 << (c << shamt);
}

// CHECK-LABEL: define i32 @f2
int f2(long c, int shamt) {
// CHECK: icmp ule i32 %{{.*}}, 63, !nosanitize
// CHECK: icmp ule i64 %{{.*}}, 31, !nosanitize
  return 1 << (c << shamt);
}

// CHECK-LABEL: define i32 @f3
unsigned f3(unsigned c, int shamt) {
// CHECK: icmp ule i32 %{{.*}}, 31, !nosanitize
// CHECK: icmp ule i32 %{{.*}}, 31, !nosanitize
  return 1U << (c << shamt);
}

// CHECK-LABEL: define i32 @f4
unsigned f4(unsigned long c, int shamt) {
// CHECK: icmp ule i32 %{{.*}}, 63, !nosanitize
// CHECK: icmp ule i64 %{{.*}}, 31, !nosanitize
  return 1U << (c << shamt);
}

// CHECK-LABEL: define i32 @f5
int f5(int c, long long shamt) {
// CHECK: icmp ule i64 %{{[0-9]+}}, 31, !nosanitize
//
// CHECK: sub nuw nsw i32 31, %sh_prom, !nosanitize
// CHECK: lshr i32 %{{.*}}, %shl.zeros, !nosanitize
  return c << shamt;
}

// CHECK-LABEL: define i32 @f6
int f6(int c, int shamt) {
// CHECK: icmp ule i32 %[[WIDTH:.*]], 31, !nosanitize
//
// CHECK: sub nuw nsw i32 31, %[[WIDTH]], !nosanitize
// CHECK: lshr i32 %{{.*}}, %shl.zeros, !nosanitize
  return c << shamt;
}
