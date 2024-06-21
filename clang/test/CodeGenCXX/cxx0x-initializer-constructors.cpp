// RUN: %clang_cc1 -std=c++11 -triple x86_64 -emit-llvm -o - %s | FileCheck %s

struct S {
  S(int x) { }
  S(int x, double y, double z) { }
};

void fn1() {
  // CHECK-LABEL: define{{.*}} void @_Z3fn1v
  S s { 1 };
  // CHECK: alloca %struct.S, align 1
  // CHECK: call void @_ZN1SC1Ei(ptr {{[^,]*}} %s, i32 noundef 1)
}

void fn2() {
  // CHECK-LABEL: define{{.*}} void @_Z3fn2v
  S s { 1, 2.0, 3.0 };
  // CHECK: alloca %struct.S, align 1
  // CHECK: call void @_ZN1SC1Eidd(ptr {{[^,]*}} %s, i32 noundef 1, double noundef 2.000000e+00, double noundef 3.000000e+00)
}

void fn3() {
  // CHECK-LABEL: define{{.*}} void @_Z3fn3v
  S sa[] { { 1 }, { 2 }, { 3 } };
  // CHECK: alloca [3 x %struct.S], align 1
  // CHECK: call void @_ZN1SC1Ei(ptr {{[^,]*}} %{{.+}}, i32 noundef 1)
  // CHECK: call void @_ZN1SC1Ei(ptr {{[^,]*}} %{{.+}}, i32 noundef 2)
  // CHECK: call void @_ZN1SC1Ei(ptr {{[^,]*}} %{{.+}}, i32 noundef 3)
}

void fn4() {
  // CHECK-LABEL: define{{.*}} void @_Z3fn4v
  S sa[] { { 1, 2.0, 3.0 }, { 4, 5.0, 6.0 } };
  // CHECK: alloca [2 x %struct.S], align 1
  // CHECK: call void @_ZN1SC1Eidd(ptr {{[^,]*}} %{{.+}}, i32 noundef 1, double noundef 2.000000e+00, double noundef 3.000000e+00)
  // CHECK: call void @_ZN1SC1Eidd(ptr {{[^,]*}} %{{.+}}, i32 noundef 4, double noundef 5.000000e+00, double noundef 6.000000e+00)
}

namespace TreeTransformBracedInit {
  struct S {};
  struct T { T(const S &); T(const T&); ~T(); };
  void x(const T &);
  template<typename> void foo(const S &s) {
    // Instantiation of this expression used to lose the CXXBindTemporaryExpr
    // node and thus not destroy the temporary.
    x({s});
  }
  template void foo<void>(const S&);
  // CHECK: define {{.*}} void @_ZN23TreeTransformBracedInit3fooIvEEvRKNS_1SE(
  // CHECK: call void @_ZN23TreeTransformBracedInit1TC1ERKNS_1SE(
  // CHECK-NEXT: call void @_ZN23TreeTransformBracedInit1xERKNS_1TE(
  // CHECK-NEXT: call void @_ZN23TreeTransformBracedInit1TD1Ev(
}
