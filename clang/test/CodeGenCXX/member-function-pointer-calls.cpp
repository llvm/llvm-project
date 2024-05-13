// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -O3 -o - | FileCheck %s
/// Check that we pass the member pointers indirectly for MinGW64
// RUN: %clang_cc1 %s -triple=x86_64-windows-gnu -emit-llvm -o - | FileCheck %s -check-prefix MINGW64
/// We should be able to optimize calls via the indirectly passed member pointers
// RUN: %clang_cc1 %s -triple=x86_64-windows-gnu -emit-llvm -O3 -o - | FileCheck %s
struct A {
  virtual int vf1() { return 1; }
  virtual int vf2() { return 2; }
};

int f(A* a, int (A::*fp)()) {
  return (a->*fp)();
}

// CHECK-LABEL: define{{.*}} i32 @_Z2g1v()
// CHECK-NOT: }
// CHECK: ret i32 1
// MINGW64-LABEL: define dso_local noundef i32 @_Z2g1v()
// MINGW64: call noundef i32 @_Z1fP1AMS_FivE(ptr noundef %{{.*}}, ptr noundef %{{.*}})
int g1() {
  A a;
  return f(&a, &A::vf1);
}

// CHECK-LABEL: define{{.*}} i32 @_Z2g2v()
// CHECK-NOT: }
// CHECK: ret i32 2
// MINGW64-LABEL: define dso_local noundef i32 @_Z2g2v()
// MINGW64: call noundef i32 @_Z1fP1AMS_FivE(ptr noundef %{{.*}}, ptr noundef %{{.*}})
int g2() {
  A a;
  return f(&a, &A::vf2);
}
