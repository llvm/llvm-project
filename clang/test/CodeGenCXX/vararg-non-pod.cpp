// RUN: %clang_cc1 -Wno-error=non-pod-varargs -emit-llvm -o - %s | FileCheck %s

struct X {
  X();
  X(const X&);
  ~X();
};

void vararg(...);

// CHECK: define void @_Z4test1X
void test(X x) {
  // CHECK: call void @llvm.trap()
  vararg(x);
  // CHECK: ret void
}
