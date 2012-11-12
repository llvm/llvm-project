// RUN: %clang_cc1 -triple i386-apple-darwin10 -mms-bitfields -emit-llvm %s -o - | FileCheck %s

struct s1 {
  int       f32;
  long long f64;
} s1;

// CHECK: %struct.s1 = type { i32, [4 x i8], i64 }

struct s2 {
  int       f32;
  long long f64[4];
} s2;

// CHECK: %struct.s2 = type { i32, [4 x i8], [4 x i64] }

struct s3 {
  int       f32;
  struct s1 s;
} s3;

// CHECK: %struct.s3 = type { i32, [4 x i8], %struct.s1 }
