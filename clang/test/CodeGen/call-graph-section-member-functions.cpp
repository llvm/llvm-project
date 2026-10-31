// Member functions with an implicit object parameter, and the virtual and
// pointer-to-member calls that reach them, get type identifiers separate from
// free functions with the same signature. An ordinary function pointer cannot
// hold such a member function, and a virtual or pointer-to-member call cannot
// reach a free function, so the split loses no edge. Static member functions,
// member functions with an explicit object parameter and the function a
// captureless lambda converts to are free functions.

// RUN: %clang_cc1 -triple aarch64-unknown-fuchsia -std=c++23 -emit-llvm -o - \
// RUN:   -fexperimental-call-graph-section %s | FileCheck %s

struct A {
  virtual void f();
  virtual ~A();
  void g();
  static void s();
};
struct B {
  virtual void f();
  virtual ~B();
};
struct C : A, B {
  void f() override;
  ~C() override;
};
struct E {
  void h(this E &);
};

// CHECK-DAG: define {{.*}} @_ZN1A1fEv({{.*}} !callgraph [[MEMBER:![0-9]+]]
// CHECK-DAG: define {{.*}} @_ZN1AD0Ev({{.*}} !callgraph [[MEMBER]]
// CHECK-DAG: define {{.*}} @_ZN1A1gEv({{.*}} !callgraph [[MEMBER]]
// CHECK-DAG: define {{.*}} @_ZN1A1sEv({{.*}} !callgraph [[FREE:![0-9]+]]
// CHECK-DAG: define {{.*}} @_ZThn8_N1C1fEv({{.*}} !callgraph [[MEMBER]]
// CHECK-DAG: define {{.*}} @_Z6freefnv({{.*}} !callgraph [[FREE]]
// CHECK-DAG: define {{.*}} @_ZNH1E1hERS_({{.*}} !callgraph [[EXPLICIT:![0-9]+]]
void A::f() {}
A::~A() {}
void A::g() {}
void A::s() {}
void B::f() {}
B::~B() {}
void C::f() {}
C::~C() {}
void freefn() {}
void E::h(this E &) {}

void (*volatile FreeFnPtr)();

// CHECK-LABEL: define {{.*}} @_Z3useP1AP1BMS_FvvER1E(
void use(A *a, B *b, void (A::*pmf)(), E &e) {
  // CHECK: call void %{{.*}}, !callee_type [[MEMBER_CT:![0-9]+]]
  a->f();
  // CHECK: call void %{{.*}}, !callee_type [[MEMBER_CT]]
  (a->*pmf)();
  // CHECK: call void %{{.*}}, !callee_type [[FREE_CT:![0-9]+]]
  FreeFnPtr();
  void (*sp)() = &A::s;
  // CHECK: call void %{{.*}}, !callee_type [[FREE_CT]]
  sp();
  auto l = [] {};
  void (*lp)() = l;
  // CHECK: call void %{{.*}}, !callee_type [[FREE_CT]]
  lp();
  // CHECK: call void %{{.*}}, !callee_type [[MEMBER_CT]]
  b->f();
  void (*ep)(E &) = &E::h;
  // CHECK: call void %{{.*}}, !callee_type [[EXPLICIT_CT:![0-9]+]]
  ep(e);
}

// CHECK-DAG: [[MEMBER]] = !{!"_ZTSFvvE.member"}
// CHECK-DAG: [[FREE]] = !{!"_ZTSFvvE"}
// CHECK-DAG: [[MEMBER_CT]] = !{[[MEMBER]]}
// CHECK-DAG: [[FREE_CT]] = !{[[FREE]]}
// CHECK-DAG: [[EXPLICIT]] = !{!"_ZTSFvR1EE"}
// CHECK-DAG: [[EXPLICIT_CT]] = !{[[EXPLICIT]]}
