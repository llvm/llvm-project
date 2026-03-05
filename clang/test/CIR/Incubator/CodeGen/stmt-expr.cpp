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
// CHECK:   %[[#RETVAL:]] = cir.alloca !rec_A, !cir.ptr<!rec_A>
// CHECK:   cir.scope {
// CHECK:     %[[#VAR:]] = cir.alloca !rec_A, !cir.ptr<!rec_A>, ["a", init] {alignment = 4 : i64}
// CHECK:     cir.call @_ZN1AC1Ev(%[[#VAR]]) : (!cir.ptr<!rec_A>) -> ()
// CHECK:     cir.copy %[[#VAR]] to %[[#RETVAL]] : !cir.ptr<!rec_A>
//            TODO(cir): the local VAR should be destroyed here.
// CHECK:   }
// CHECK:   cir.call @_ZN1A3FooEv(%[[#RETVAL]]) : (!cir.ptr<!rec_A>) -> ()
//          TODO(cir): the temporary RETVAL should be destroyed here.
// CHECK: }
