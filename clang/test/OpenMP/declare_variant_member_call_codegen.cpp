// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

// Check calls to member functions with a declare variant. The variant must be
// called through the same object expression, and with the same '.' or '->', as
// the base function, including for calls that are not CXXMemberCallExprs:
// calls to static member functions, and member calls in templates.

struct S {
  void m_v(int);
#pragma omp declare variant(m_v) match(user = {condition(1)})
  void m(int);

  static void s_v(int);
#pragma omp declare variant(s_v) match(user = {condition(1)})
  static void s(int);

  void implicit_this(int a) { m(a); }
  void static_in_member(int a) { s(a); }
};

// CHECK-LABEL: define {{.*}}void @_Z7via_objR1Si(
// CHECK: [[OBJ:%.+]] = load ptr, ptr %o.addr
// CHECK: call void @_ZN1S3m_vEi(ptr {{.*}}[[OBJ]], i32
void via_obj(S &o, int a) { o.m(a); }

// The pointer, not the address of the variable holding it, is 'this'.
// CHECK-LABEL: define {{.*}}void @_Z7via_ptrP1Si(
// CHECK: [[PTR:%.+]] = load ptr, ptr %p.addr
// CHECK: call void @_ZN1S3m_vEi(ptr {{.*}}[[PTR]], i32
void via_ptr(S *p, int a) { p->m(a); }

// CHECK-LABEL: define {{.*}}void @_Z11static_quali(
// CHECK: call void @_ZN1S3s_vEi(i32
void static_qual(int a) { S::s(a); }

// The object expression of a static member call is still evaluated.
S &get();
// CHECK-LABEL: define {{.*}}void @_Z14static_via_obji(
// CHECK: call {{.*}} @_Z3getv()
// CHECK: call void @_ZN1S3s_vEi(i32
void static_via_obj(int a) { get().s(a); }

template <typename T> struct CT {
  void m_v(T);
#pragma omp declare variant(m_v) match(user = {condition(1)})
  void m(T);

  void n_v(int);
#pragma omp declare variant(n_v) match(user = {condition(1)})
  void n(int);

  static void s_v(T);
#pragma omp declare variant(s_v) match(user = {condition(1)})
  static void s(T);

  void dependent(T t) { m(t); }
  void dependent_arrow(T t) { this->m(t); }
  void nondependent(int a) { n(a); }
  void dependent_static(T t) { s(t); }
};

template <typename T> void via_ptr_tmpl(CT<T> *p, T t) { p->m(t); }

void members(S &o, CT<int> &ci, CT<long> &cl, int a) {
  o.implicit_this(a);
  o.static_in_member(a);
  ci.dependent(a);
  cl.dependent(a);
  ci.dependent_arrow(a);
  ci.nondependent(a);
  ci.dependent_static(a);
  via_ptr_tmpl(&cl, (long)a);
}

// CHECK-LABEL: define {{.*}}void @_ZN1S13implicit_thisEi(
// CHECK: [[THIS:%.+]] = load ptr, ptr %this.addr
// CHECK: call void @_ZN1S3m_vEi(ptr {{.*}}[[THIS]], i32

// CHECK-LABEL: define {{.*}}void @_ZN1S16static_in_memberEi(
// CHECK: call void @_ZN1S3s_vEi(i32

// CHECK-LABEL: define {{.*}}void @_ZN2CTIiE9dependentEi(
// CHECK: [[THIS:%.+]] = load ptr, ptr %this.addr
// CHECK: call void @_ZN2CTIiE3m_vEi(ptr {{.*}}[[THIS]], i32

// CHECK-LABEL: define {{.*}}void @_ZN2CTIlE9dependentEl(
// CHECK: [[THIS:%.+]] = load ptr, ptr %this.addr
// CHECK: call void @_ZN2CTIlE3m_vEl(ptr {{.*}}[[THIS]], i64

// CHECK-LABEL: define {{.*}}void @_ZN2CTIiE15dependent_arrowEi(
// CHECK: [[THIS:%.+]] = load ptr, ptr %this.addr
// CHECK: call void @_ZN2CTIiE3m_vEi(ptr {{.*}}[[THIS]], i32

// CHECK-LABEL: define {{.*}}void @_ZN2CTIiE12nondependentEi(
// CHECK: [[THIS:%.+]] = load ptr, ptr %this.addr
// CHECK: call void @_ZN2CTIiE3n_vEi(ptr {{.*}}[[THIS]], i32

// CHECK-LABEL: define {{.*}}void @_ZN2CTIiE16dependent_staticEi(
// CHECK: call void @_ZN2CTIiE3s_vEi(i32

// CHECK-LABEL: define {{.*}}void @_Z12via_ptr_tmplIlEvP2CTIT_ES1_(
// CHECK: [[PTR:%.+]] = load ptr, ptr %p.addr
// CHECK: call void @_ZN2CTIlE3m_vEl(ptr {{.*}}[[PTR]], i64
