// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -triple aarch64-none-elf \
// RUN:   -O2 \
// RUN:   -emit-llvm -o - %s | FileCheck %s

extern "C" {

// Base case, nothing interesting.
struct S {
  long x, y;
};

void f0(long, S);
void f0m(long, long, long, long, long, S);
void g0() {
  S s = {6, 7};
  f0(1, s);
  f0m(1, 2, 3, 4, 5, s);
}
// CHECK: define{{.*}} void @g0
// CHECK: call void @f0(i64 noundef 1, [2 x i64] [i64 6, i64 7]
// CHECK: call void @f0m{{.*}}[2 x i64] [i64 6, i64 7]
// CHECK: declare void @f0(i64 noundef, [2 x i64])
// CHECK: declare void @f0m(i64 noundef, i64 noundef, i64 noundef, i64 noundef, i64 noundef, [2 x i64])

// Aligned struct, passed according to its natural alignment.
struct __attribute__((aligned(16))) S16 {
  long x, y;
} s16;

void f1(long, S16);
void f1m(long, long, long, long, long, S16);
void g1() {
  S16 s = {6, 7};
  f1(1, s);
  f1m(1, 2, 3, 4, 5, s);
}
// CHECK: define{{.*}} void @g1
// CHECK: call void @f1{{.*}}[2 x i64] [i64 6, i64 7]
// CHECK: call void @f1m{{.*}}[2 x i64] [i64 6, i64 7]
// CHECK: declare void @f1(i64 noundef, [2 x i64])
// CHECK: declare void @f1m(i64 noundef, i64 noundef, i64 noundef, i64 noundef, i64 noundef, [2 x i64])

// Increased natural alignment.
struct SF16 {
  long x __attribute__((aligned(16)));
  long y;
};

void f3(long, SF16);
void f3m(long, long, long, long, long, SF16);
void g3() {
  SF16 s = {6, 7};
  f3(1, s);
  f3m(1, 2, 3, 4, 5, s);
}
// CHECK: define{{.*}} void @g3
// CHECK: call void @f3(i64 noundef 1, i128 129127208515966861318)
// CHECK: call void @f3m(i64 noundef 1, i64 noundef 2, i64 noundef 3, i64 noundef 4, i64 noundef 5, i128 129127208515966861318)
// CHECK: declare void @f3(i64 noundef, i128)
// CHECK: declare void @f3m(i64 noundef, i64 noundef, i64 noundef, i64 noundef, i64 noundef, i128)


// Packed structure.
struct  __attribute__((packed)) P {
  int x;
  long u;
};

void f4(int, P);
void f4m(int, int, int, int, int, P);
void g4() {
  P s = {6, 7};
  f4(1, s);
  f4m(1, 2, 3, 4, 5, s);
}
// CHECK: define{{.*}} void @g4()
// CHECK: call void @f4(i32 noundef 1, [2 x i64] [i64 30064771078, i64 0])
// CHECK: void @f4m(i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 4, i32 noundef 5, [2 x i64] [i64 30064771078, i64 0])
// CHECK: declare void @f4(i32 noundef, [2 x i64])
// CHECK: declare void @f4m(i32 noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, [2 x i64])


// Packed structure, overaligned, same as above.
struct  __attribute__((packed, aligned(16))) P16 {
  int x;
  long y;
};

void f5(int, P16);
void f5m(int, int, int, int, int, P16);
  void g5() {
    P16 s = {6, 7};
    f5(1, s);
    f5m(1, 2, 3, 4, 5, s);
}
// CHECK: define{{.*}} void @g5()
// CHECK: call void @f5(i32 noundef 1, [2 x i64] [i64 30064771078, i64 0])
// CHECK: void @f5m(i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 4, i32 noundef 5, [2 x i64] [i64 30064771078, i64 0])
// CHECK: declare void @f5(i32 noundef, [2 x i64])
// CHECK: declare void @f5m(i32 noundef, i32 noundef, i32 noundef, i32 noundef, i32 noundef, [2 x i64])

}
