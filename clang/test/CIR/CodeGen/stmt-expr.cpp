// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

class A {
public:
  A(): x(0) {}
  A(A &a) : x(a.x) {}
  // TODO(cir): Ensure dtors are properly called. The dtor below crashes.
  // ~A() {}
  int x;
  void Foo() {}
};

void test1() {
  ({
    A a;
    a;
  }).Foo();
}
// CHECK: @_Z5test1v
// CHECK: cir.scope {
// CHECK:   %[[#RETVAL:]] = cir.alloca !ty_22A22, cir.ptr <!ty_22A22>
// CHECK:   cir.scope {
// CHECK:     %[[#VAR:]] = cir.alloca !ty_22A22, cir.ptr <!ty_22A22>, ["a", init] {alignment = 4 : i64}
// CHECK:     cir.call @_ZN1AC1Ev(%[[#VAR]]) : (!cir.ptr<!ty_22A22>) -> ()
// CHECK:     cir.call @_ZN1AC1ERS_(%[[#RETVAL]], %[[#VAR]]) : (!cir.ptr<!ty_22A22>, !cir.ptr<!ty_22A22>) -> ()
//            TODO(cir): the local VAR should be destroyed here.
// CHECK:   }
// CHECK:   cir.call @_ZN1A3FooEv(%[[#RETVAL]]) : (!cir.ptr<!ty_22A22>) -> ()
//          TODO(cir): the temporary RETVAL should be destroyed here.
// CHECK: }
