// RUN: %clang_cc1 -emit-llvm -o - -O0 -triple x86_64-apple-darwin10 %s | FileCheck %s

// Ensure that we don't emit available_externally functions at -O0.
int x;

inline void f0(int y) { x = y; }

// CHECK: define void @test()
// CHECK: declare void @f0(i32)
void test() {
  f0(17);
}

inline int __attribute__((always_inline)) f1(int x) {
  int blarg = 0;
  for (int i = 0; i < x; ++i)
    blarg = blarg + x * i;
  return blarg;
}

// CHECK: @test1
int test1(int x) {
  // CHECK: br i1
  // CHECK-NOT: call {{.*}} @f1
  // CHECK: ret i32
  return f1(x);
}
