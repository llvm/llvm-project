// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -triple arm-none-none-eabi \
// RUN:   -O2 \
// RUN:   -target-cpu cortex-a8 \
// RUN:   -emit-llvm -o - %s | FileCheck %s

extern "C" {

// Base case, nothing interesting.
struct S {
  int x, y;
};

void f0(int, S);
void f0m(int, int, int, int, int, S);
void g0() {
  S s = {6, 7};
  f0(1, s);
  f0m(1, 2, 3, 4, 5, s);
}
// CHECK: define{{.*}} void @g0
// CHECK: call void @f0(i32 noundef 1, [2 x i32] [i32 6, i32 7]
// CHECK: call void @f0m(i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 4, i32 noundef 5, [2 x i32] [i32 6, i32 7]
// CHECK: declare void @f0(i32 noundef, [2 x i32])
// CHECK: declare void @f0m(i32 noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, [2 x i32])

// Aligned struct, passed according to its natural alignment.
struct __attribute__((aligned(8))) S8 {
  int x, y;
} s8;

void f1(int, S8);
void f1m(int, int, int, int, int, S8);
void g1() {
  S8 s = {6, 7};
  f1(1, s);
  f1m(1, 2, 3, 4, 5, s);
}
// CHECK: define{{.*}} void @g1
// CHECK: call void @f1(i32 noundef 1, [2 x i32] [i32 6, i32 7]
// CHECK: call void @f1m(i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 4, i32 noundef 5, [2 x i32] [i32 6, i32 7]
// CHECK: declare void @f1(i32 noundef, [2 x i32])
// CHECK: declare void @f1m(i32 noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, [2 x i32])

// Aligned struct, passed according to its natural alignment.
struct alignas(16) S16 {
  int x, y;
};

extern "C" void f2(int, S16);
extern "C" void f2m(int, int, int, int, int, S16);

void g2() {
  S16 s = {6, 7};
  f2(1, s);
  f2m(1, 2, 3, 4, 5, s);
}
// CHECK: define{{.*}} void @g2
// CHECK: call void @f2(i32 noundef 1, [4 x i32] [i32 6, i32 7
// CHECK: call void @f2m(i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 4, i32 noundef 5, [4 x i32] [i32 6, i32 7
// CHECK: declare void @f2(i32 noundef, [4 x i32])
// CHECK: declare void @f2m(i32 noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, [4 x i32])

// Increased natural alignment.
struct SF8 {
  int x __attribute__((aligned(8)));
  int y;
};

void f3(int, SF8);
void f3m(int, int, int, int, int, SF8);
void g3() {
  SF8 s = {6, 7};
  f3(1, s);
  f3m(1, 2, 3, 4, 5, s);
}
// CHECK: define{{.*}} void @g3
// CHECK: call void @f3(i32 noundef 1, [1 x i64] [i64 30064771078]
// CHECK: call void @f3m(i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 4, i32 noundef 5, [1 x i64] [i64 30064771078]
// CHECK: declare void @f3(i32 noundef, [1 x i64])
// CHECK: declare void @f3m(i32 noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, [1 x i64])

// Increased natural alignment, capped to 8 though.
struct SF16 {
  int x;
  int y alignas(16);
  int z, a, b, c, d, e, f, g, h, i, j, k;
};

void f4(int, SF16);
void f4m(int, int, int, int, int, SF16);
void g4() {
  SF16 s = {6, 7};
  f4(1, s);
  f4m(1, 2, 3, 4, 5, s);
}
// CHECK: define{{.*}} void @g4
// CHECK: call void @f4(i32 noundef 1, ptr noundef nonnull byval(%struct.SF16) align 8
// CHECK: call void @f4m(i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 4, i32 noundef 5, ptr noundef nonnull byval(%struct.SF16) align 8
// CHECK: declare void @f4(i32 noundef, ptr noundef byval(%struct.SF16) align 8)
// CHECK: declare void @f4m(i32 noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, ptr noundef byval(%struct.SF16) align 8)

// Packed structure.
struct  __attribute__((packed)) P {
  int x;
  long long u;
};

void f5(int, P);
void f5m(int, int, int, int, int, P);
void g5() {
  P s = {6, 7};
  f5(1, s);
  f5m(1, 2, 3, 4, 5, s);
}
// CHECK: define{{.*}} void @g5
// CHECK: call void @f5(i32 noundef 1, [3 x i32] [i32 6, i32 7, i32 0])
// CHECK: call void @f5m(i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 4, i32 noundef 5, [3 x i32] [i32 6, i32 7, i32 0])
// CHECK: declare void @f5(i32 noundef, [3 x i32])
// CHECK: declare void @f5m(i32 noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, [3 x i32])


// Packed and aligned, alignement causes padding at the end.
struct  __attribute__((packed, aligned(8))) P8 {
  int x;
  long long u;
};

void f6(int, P8);
void f6m(int, int, int, int, int, P8);
void g6() {
  P8 s = {6, 7};
  f6(1, s);
  f6m(1, 2, 3, 4, 5, s);
}
// CHECK: define{{.*}} void @g6
// CHECK: call void @f6(i32 noundef 1, [4 x i32] [i32 6, i32 7, i32 0, i32 undef])
// CHECK: call void @f6m(i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 4, i32 noundef 5, [4 x i32] [i32 6, i32 7, i32 0, i32 undef])
// CHECK: declare void @f6(i32 noundef, [4 x i32])
// CHECK: declare void @f6m(i32 noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, [4 x i32])
}
