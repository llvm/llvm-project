// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s

struct s0 {
  int Start, End;
  unsigned Alignment;
  int TheStores __attribute__((aligned(16)));
};

// CHECK: define void @f0
// CHECK: alloca %struct.s0, align 16
extern "C" void f0() {
  (void) s0();
}

// CHECK: define void @f1
// CHECK: alloca %struct.s0, align 16
extern "C" void f1() {
  (void) (struct s0) { 0, 0, 0, 0 };
}

// CHECK: define i32 @f2
// CHECK: alloca %struct.s1, align 2
struct s1 { short x; short y; };
extern "C" struct s1 f2(int a, struct s1 *x, struct s1 *y) {
  if (a)
    return *x;
  return *y;
}
