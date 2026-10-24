// Tests that virtual calls through a pointer to any class of a hierarchy get
// the same call graph section type identifier as the overriders they can reach,
// even when the overriders have a covariant return type.

// RUN: %clang_cc1 -triple x86_64-unknown-linux -fexperimental-call-graph-section \
// RUN: -std=c++17 -emit-llvm -o - %s | FileCheck %s

////////////////////////////////////////////////////////////////////////////////
// A chain of covariant overrides: all methods are identified by the type of
// Base::clone, the method that introduced the virtual function.

struct Base {
  virtual ~Base() {}
  virtual Base *clone() { return this; }
};

struct Mid : Base {
  Mid *clone() override { return this; }
};

struct Derived : Mid {
  Derived *clone() override { return this; }
};

////////////////////////////////////////////////////////////////////////////////
// A method overriding two virtual functions with different covariant return
// types: the method is identified after the first one (A::f), and the return
// adjusting thunk that fills the vtable slot of B::f after the second one.

struct A {
  virtual A *f() { return this; }
};
struct B {
  virtual B *f() { return this; }
};
struct C : A, B {
  C *f() override { return this; }
};

////////////////////////////////////////////////////////////////////////////////
// Call sites.

// CHECK-LABEL: define {{.*}} @_Z12call_clone_bP4Base(
// CHECK: call noundef ptr %{{.*}}, !callee_type [[CLONE_CT:![0-9]+]]
Base *call_clone_b(Base *b) { return b->clone(); }

// CHECK-LABEL: define {{.*}} @_Z12call_clone_mP3Mid(
// CHECK: call noundef ptr %{{.*}}, !callee_type [[CLONE_CT]]
Mid *call_clone_m(Mid *m) { return m->clone(); }

// CHECK-LABEL: define {{.*}} @_Z12call_clone_dP7Derived(
// CHECK: call noundef ptr %{{.*}}, !callee_type [[CLONE_CT]]
Derived *call_clone_d(Derived *d) { return d->clone(); }

// CHECK-LABEL: define {{.*}} @_Z8call_f_aP1A(
// CHECK: call noundef ptr %{{.*}}, !callee_type [[A_F_CT:![0-9]+]]
A *call_f_a(A *a) { return a->f(); }

// CHECK-LABEL: define {{.*}} @_Z8call_f_bP1B(
// CHECK: call noundef ptr %{{.*}}, !callee_type [[B_F_CT:![0-9]+]]
B *call_f_b(B *b) { return b->f(); }

// CHECK-LABEL: define {{.*}} @_Z8call_f_cP1C(
// CHECK: call noundef ptr %{{.*}}, !callee_type [[A_F_CT]]
C *call_f_c(C *c) { return c->f(); }

// Instantiate the vtables (and thereby the methods and thunks below).
Base *make() { return new Derived(); }
C *make_c() { return new C(); }

////////////////////////////////////////////////////////////////////////////////
// Indirect call targets (emitted after the functions above).

// CHECK-DAG: define {{.*}} @_ZN7Derived5cloneEv({{.*}} !callgraph [[CLONE:![0-9]+]] {
// CHECK-DAG: define {{.*}} @_ZN3Mid5cloneEv({{.*}} !callgraph [[CLONE]] {
// CHECK-DAG: define {{.*}} @_ZN4Base5cloneEv({{.*}} !callgraph [[CLONE]] {
// CHECK-DAG: define {{.*}} @_ZN1C1fEv({{.*}} !callgraph [[A_F:![0-9]+]] {
// CHECK-DAG: define {{.*}} @_ZTchn8_h8_N1C1fEv({{.*}} !callgraph [[B_F:![0-9]+]] {
// CHECK-DAG: define {{.*}} @_ZN1A1fEv({{.*}} !callgraph [[A_F]] {
// CHECK-DAG: define {{.*}} @_ZN1B1fEv({{.*}} !callgraph [[B_F]] {

// CHECK-DAG: [[CLONE]] = !{!"_ZTSFP4BasevE"}
// CHECK-DAG: [[CLONE_CT]] = !{[[CLONE]]}
// CHECK-DAG: [[A_F]] = !{!"_ZTSFP1AvE"}
// CHECK-DAG: [[A_F_CT]] = !{[[A_F]]}
// CHECK-DAG: [[B_F]] = !{!"_ZTSFP1BvE"}
// CHECK-DAG: [[B_F_CT]] = !{[[B_F]]}
