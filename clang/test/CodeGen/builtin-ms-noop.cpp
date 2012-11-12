// RUN: %clang_cc1 -triple i686-pc-win32 -emit-llvm %s -o - | FileCheck %s

class A {
 public:
  ~A() {}
};

void f() {
// CHECK: @_Z1fv
// CHECK-NOT: call void @_ZN1AD1Ev
// CHECK: ret void
  __noop(A());
};

