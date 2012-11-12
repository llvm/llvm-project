// RUN: %clang_cc1 -triple i386-apple-darwin9 %s -emit-llvm -o - | FileCheck -check-prefix X32 %s
// CHECK-X32: %struct.s0 = type { i64, i64, i32, [12 x i32] }
// CHECK-X32: %struct.s1 = type { [15 x i32], %struct.s0 }

// RUN: %clang_cc1 -triple x86_64-apple-darwin9 %s -emit-llvm -o - | FileCheck -check-prefix X64 %s
// CHECK-X64: %struct.s0 = type <{ i64, i64, i32, [12 x i32] }>
// CHECK-X64: %struct.s1 = type <{ [15 x i32], %struct.s0 }>

// rdar://problem/7095436
#pragma pack(4)

struct s0 {
  long long a __attribute__((aligned(8)));
  long long b __attribute__((aligned(8)));
  unsigned int c __attribute__((aligned(8)));
  int d[12];
} a;

struct s1 {
  int a[15];
  struct s0 b;
} b;

