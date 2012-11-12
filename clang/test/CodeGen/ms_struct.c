// RUN: %clang_cc1 -triple i386-apple-darwin10 -emit-llvm %s -o - | FileCheck %s

#define ATTR __attribute__((__ms_struct__))
struct s1 {
  int       f32;
  long long f64;
} ATTR s1;

// CHECK: %struct.s1 = type { i32, [4 x i8], i64 }

struct s2 {
  int       f32;
  long long f64[4];
} ATTR s2;

// CHECK: %struct.s2 = type { i32, [4 x i8], [4 x i64] }

struct s3 {
  int       f32;
  struct s1 s;
} ATTR s3;

// CHECK: %struct.s3 = type { i32, [4 x i8], %struct.s1 }
