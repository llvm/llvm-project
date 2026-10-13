// Tests that virtual calls through a pointer to any class of a hierarchy get
// the same call graph section type identifier as the functions they can reach
// through the vtable entry they use, even when overriders have a covariant
// return type.

// RUN: %clang_cc1 -triple x86_64-unknown-linux -fexperimental-call-graph-section \
// RUN: -std=c++17 -emit-llvm -o - %s | FileCheck %s

////////////////////////////////////////////////////////////////////////////////
// A chain of covariant overrides: every method is placed in the vtable entry of
// Base::clone and identified after it.

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

// CHECK-LABEL: define {{.*}} @_Z12call_clone_bP4Base(
// CHECK: call noundef ptr %{{.*}}, !callee_type [[CLONE_CT:![0-9]+]]
Base *call_clone_b(Base *b) { return b->clone(); }

// CHECK-LABEL: define {{.*}} @_Z12call_clone_mP3Mid(
// CHECK: call noundef ptr %{{.*}}, !callee_type [[CLONE_CT]]
Mid *call_clone_m(Mid *m) { return m->clone(); }

// CHECK-LABEL: define {{.*}} @_Z12call_clone_dP7Derived(
// CHECK: call noundef ptr %{{.*}}, !callee_type [[CLONE_CT]]
Derived *call_clone_d(Derived *d) { return d->clone(); }

Base *make() { return new Derived(); }


////////////////////////////////////////////////////////////////////////////////
// A method overriding two virtual functions: C::f is placed in the entry of
// A::f, the method it overrides in its primary base, and identified after it.
// The entry of B::f, in the vtable of the non-primary base B, is filled by a
// thunk identified after B::f.

struct A {
  virtual A *f() { return this; }
};
struct B {
  virtual B *f() { return this; }
};
struct C : A, B {
  C *f() override { return this; }
};

// CHECK-LABEL: define {{.*}} @_Z8call_f_aP1A(
// CHECK: call noundef ptr %{{.*}}, !callee_type [[A_F_CT:![0-9]+]]
A *call_f_a(A *a) { return a->f(); }

// CHECK-LABEL: define {{.*}} @_Z8call_f_bP1B(
// CHECK: call noundef ptr %{{.*}}, !callee_type [[B_F_CT:![0-9]+]]
B *call_f_b(B *b) { return b->f(); }

// CHECK-LABEL: define {{.*}} @_Z8call_f_cP1C(
// CHECK: call noundef ptr %{{.*}}, !callee_type [[A_F_CT]]
C *call_f_c(C *c) { return c->f(); }

C *make_c() { return new C(); }


////////////////////////////////////////////////////////////////////////////////
// The same, without any return adjustment: A1 and B1 are both at offset zero
// in C1. The entry of B2::f is still filled by a (this adjusting) thunk.

namespace no_return_adjustment {
struct A1 {};
struct B1 {};
struct C1 : A1, B1 {};
struct A2 {
  virtual A1 *f() { return nullptr; }
};
struct B2 {
  virtual B1 *f() { return nullptr; }
};
struct C2 : A2, B2 {
  C1 *f() override { return nullptr; }
};

// CHECK-LABEL: define {{.*}} @_ZN20no_return_adjustment9call_f_a2EPNS_2A2E(
// CHECK: call noundef ptr %{{.*}}, !callee_type [[NRA_A_CT:![0-9]+]]
A1 *call_f_a2(A2 *a) { return a->f(); }

// CHECK-LABEL: define {{.*}} @_ZN20no_return_adjustment9call_f_b2EPNS_2B2E(
// CHECK: call noundef ptr %{{.*}}, !callee_type [[NRA_B_CT:![0-9]+]]
B1 *call_f_b2(B2 *b) { return b->f(); }

// CHECK-LABEL: define {{.*}} @_ZN20no_return_adjustment9call_f_c2EPNS_2C2E(
// CHECK: call noundef ptr %{{.*}}, !callee_type [[NRA_A_CT]]
C1 *call_f_c2(C2 *c) { return c->f(); }

C2 c;

} // namespace no_return_adjustment

////////////////////////////////////////////////////////////////////////////////
// A return adjustment in the primary base: C2::f gets an entry of its own in
// the vtable of C2, and the entry of A2::f is filled by a return adjusting
// thunk. Both are identified after A2::f, like the calls through C2 and A2
// pointers that reach them.

namespace primary_return_adjustment {
struct A1 {
  int a1;
};
struct B1 {
  int b1;
};
struct C1 : B1, A1 {};
struct A2 {
  virtual A1 *f() { return nullptr; }
};
struct B2 {
  virtual B1 *f() { return nullptr; }
};
struct C2 : A2, B2 {
  C1 *f() override { return nullptr; }
};

// CHECK-LABEL: define {{.*}} @_ZN25primary_return_adjustment9call_f_a2EPNS_2A2E(
// CHECK: call noundef ptr %{{.*}}, !callee_type [[PRA_A_CT:![0-9]+]]
A1 *call_f_a2(A2 *a) { return a->f(); }

// CHECK-LABEL: define {{.*}} @_ZN25primary_return_adjustment9call_f_b2EPNS_2B2E(
// CHECK: call noundef ptr %{{.*}}, !callee_type [[PRA_B_CT:![0-9]+]]
B1 *call_f_b2(B2 *b) { return b->f(); }

// CHECK-LABEL: define {{.*}} @_ZN25primary_return_adjustment9call_f_c2EPNS_2C2E(
// CHECK: call noundef ptr %{{.*}}, !callee_type [[PRA_A_CT]]
C1 *call_f_c2(C2 *c) { return c->f(); }

C2 c;

} // namespace primary_return_adjustment

////////////////////////////////////////////////////////////////////////////////
// The primary base is not the first base: B2 is the first non-virtual dynamic
// base of C2, so C2::f is placed in the entry of B2::f and identified after it,
// although A2::f comes first among the methods it overrides.

namespace virtual_base_first {
struct A1 {
  int a1;
};
struct B1 {
  int b1;
};
struct C1 : B1, A1 {};
struct A2 {
  virtual A1 *f() { return nullptr; }
};
struct B2 {
  virtual B1 *f() { return nullptr; }
};
struct C2 : virtual A2, B2 {
  C1 *f() override { return nullptr; }
};

// CHECK-LABEL: define {{.*}} @_ZN18virtual_base_first9call_f_a2EPNS_2A2E(
// CHECK: call noundef ptr %{{.*}}, !callee_type [[VBF_A_CT:![0-9]+]]
A1 *call_f_a2(A2 *a) { return a->f(); }

// CHECK-LABEL: define {{.*}} @_ZN18virtual_base_first9call_f_b2EPNS_2B2E(
// CHECK: call noundef ptr %{{.*}}, !callee_type [[VBF_B_CT:![0-9]+]]
B1 *call_f_b2(B2 *b) { return b->f(); }

// CHECK-LABEL: define {{.*}} @_ZN18virtual_base_first9call_f_c2EPNS_2C2E(
// CHECK: call noundef ptr %{{.*}}, !callee_type [[VBF_B_CT]]
C1 *call_f_c2(C2 *c) { return c->f(); }

C2 c;

} // namespace virtual_base_first

////////////////////////////////////////////////////////////////////////////////
// Indirect call targets (emitted after the functions above).

// CHECK-DAG: define {{.*}} @_ZN7Derived5cloneEv({{.*}} !callgraph [[CLONE:![0-9]+]] {
// CHECK-DAG: define {{.*}} @_ZN3Mid5cloneEv({{.*}} !callgraph [[CLONE]] {
// CHECK-DAG: define {{.*}} @_ZN4Base5cloneEv({{.*}} !callgraph [[CLONE]] {
// CHECK-DAG: define {{.*}} @_ZN1C1fEv({{.*}} !callgraph [[A_F:![0-9]+]] {
// CHECK-DAG: define {{.*}} @_ZTchn8_h8_N1C1fEv({{.*}} !callgraph [[B_F:![0-9]+]] {
// CHECK-DAG: define {{.*}} @_ZN1A1fEv({{.*}} !callgraph [[A_F]] {
// CHECK-DAG: define {{.*}} @_ZN1B1fEv({{.*}} !callgraph [[B_F]] {
// CHECK-DAG: define {{.*}} @_ZN20no_return_adjustment2C21fEv({{.*}} !callgraph [[NRA_A:![0-9]+]] {
// CHECK-DAG: define {{.*}} @_ZThn8_N20no_return_adjustment2C21fEv({{.*}} !callgraph [[NRA_B:![0-9]+]] {
// CHECK-DAG: define {{.*}} @_ZN25primary_return_adjustment2C21fEv({{.*}} !callgraph [[PRA_A:![0-9]+]] {
// CHECK-DAG: define {{.*}} @_ZTch0_h4_N25primary_return_adjustment2C21fEv({{.*}} !callgraph [[PRA_A]] {
// CHECK-DAG: define {{.*}} @_ZThn8_N25primary_return_adjustment2C21fEv({{.*}} !callgraph [[PRA_B:![0-9]+]] {
// CHECK-DAG: define {{.*}} @_ZN18virtual_base_first2C21fEv({{.*}} !callgraph [[VBF_B:![0-9]+]] {
// CHECK-DAG: define {{.*}} @_ZTcv0_n24_h4_N18virtual_base_first2C21fEv({{.*}} !callgraph [[VBF_A:![0-9]+]] {

// CHECK-DAG: [[CLONE]] = !{!"_ZTSFP4BasevE"}
// CHECK-DAG: [[CLONE_CT]] = !{[[CLONE]]}
// CHECK-DAG: [[A_F]] = !{!"_ZTSFP1AvE"}
// CHECK-DAG: [[A_F_CT]] = !{[[A_F]]}
// CHECK-DAG: [[B_F]] = !{!"_ZTSFP1BvE"}
// CHECK-DAG: [[B_F_CT]] = !{[[B_F]]}
// CHECK-DAG: [[NRA_A]] = !{!"_ZTSFPN20no_return_adjustment2A1EvE"}
// CHECK-DAG: [[NRA_A_CT]] = !{[[NRA_A]]}
// CHECK-DAG: [[NRA_B]] = !{!"_ZTSFPN20no_return_adjustment2B1EvE"}
// CHECK-DAG: [[NRA_B_CT]] = !{[[NRA_B]]}
// CHECK-DAG: [[PRA_A]] = !{!"_ZTSFPN25primary_return_adjustment2A1EvE"}
// CHECK-DAG: [[PRA_A_CT]] = !{[[PRA_A]]}
// CHECK-DAG: [[PRA_B]] = !{!"_ZTSFPN25primary_return_adjustment2B1EvE"}
// CHECK-DAG: [[PRA_B_CT]] = !{[[PRA_B]]}
// CHECK-DAG: [[VBF_A]] = !{!"_ZTSFPN18virtual_base_first2A1EvE"}
// CHECK-DAG: [[VBF_A_CT]] = !{[[VBF_A]]}
// CHECK-DAG: [[VBF_B]] = !{!"_ZTSFPN18virtual_base_first2B1EvE"}
// CHECK-DAG: [[VBF_B_CT]] = !{[[VBF_B]]}
