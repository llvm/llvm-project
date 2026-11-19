// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-unknown -include-pch %t -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

// Check the code generated for the 'novariants' and 'nocontext' clauses of the
// dispatch construct, in forms not covered by dispatch_codegen.cpp: clauses
// that select nothing, template instantiation, member function variants, and
// round-tripping the alternative calls through a PCH.

#ifndef HEADER
#define HEADER

void f_disp(int);
void f_noctx(int);
#pragma omp declare variant(f_disp) match(construct={dispatch})
#pragma omp declare variant(f_noctx) match(user={condition(1)})
void f(int);

int x;

// The target-call must still be emitted when the only clauses are ones that do
// not select a variant.
// CHECK-LABEL: define {{.*}}void @_Z11only_dependi(
// CHECK: call void @__kmpc_omp_taskwait_deps_51(
// CHECK-NEXT: load i32, ptr %a.addr
// CHECK-NEXT: call void @_Z6f_dispi(
// CHECK-NEXT: ret void
void only_depend(int a) {
#pragma omp dispatch depend(out:x)
  f(a);
}

// CHECK-LABEL: define {{.*}}void @_Z11only_nowaiti(
// CHECK: call void @_Z6f_dispi(
// CHECK-NEXT: ret void
void only_nowait(int a) {
#pragma omp dispatch nowait
  f(a);
}

// Each clause tests its own condition. 'nocontext' is only reached when
// 'novariants' is false.
// CHECK-LABEL: define {{.*}}void @_Z4bothiii(
// CHECK: load i32, ptr %a.addr
// CHECK: store i8 %{{.+}}, ptr [[ACAP:%.capture_expr.[0-9]*]]
// CHECK: load i32, ptr %b.addr
// CHECK: store i8 %{{.+}}, ptr [[BCAP:%.capture_expr.[0-9]*]]
// CHECK: load i8, ptr [[ACAP]]
// CHECK: br i1 %{{.+}}, label %[[NVTHEN:.+]], label %[[NVELSE:.+]]
// CHECK: [[NVTHEN]]:
// CHECK-NEXT: load i32, ptr %v.addr
// CHECK-NEXT: call void @_Z1fi(
// CHECK: [[NVELSE]]:
// CHECK-NEXT: load i8, ptr [[BCAP]]
// CHECK: br i1 %{{.+}}, label %[[NCTHEN:.+]], label %[[NCELSE:.+]]
// CHECK: [[NCTHEN]]:
// CHECK-NEXT: load i32, ptr %v.addr
// CHECK-NEXT: call void @_Z7f_noctxi(
// CHECK: [[NCELSE]]:
// CHECK-NEXT: load i32, ptr %v.addr
// CHECK-NEXT: call void @_Z6f_dispi(
void both(int a, int b, int v) {
#pragma omp dispatch novariants(a) nocontext(b)
  f(v);
}

struct S {
  void m_disp(int);
  void m_noctx(int);
#pragma omp declare variant(m_disp) match(construct={dispatch})
#pragma omp declare variant(m_noctx) match(user={condition(1)})
  void m(int);

  static void s_disp(int);
#pragma omp declare variant(s_disp) match(construct={dispatch})
  static void s(int);
};

// CHECK-LABEL: define {{.*}}void @_Z6memberR1Sibb(
// CHECK: br i1
// CHECK: call void @_ZN1S1mEi(ptr {{.*}}, i32
// CHECK: br i1
// CHECK: call void @_ZN1S7m_noctxEi(ptr {{.*}}, i32
// CHECK: call void @_ZN1S6m_dispEi(ptr {{.*}}, i32
void member(S &obj, int a, bool c1, bool c2) {
#pragma omp dispatch novariants(c1) nocontext(c2)
  obj.m(a);
}

// CHECK-LABEL: define {{.*}}void @_Z12staticmemberib(
// CHECK: br i1
// CHECK: call void @_ZN1S1sEi(i32
// CHECK: call void @_ZN1S6s_dispEi(i32
void staticmember(int a, bool c) {
#pragma omp dispatch novariants(c)
  S::s(a);
}

// 'nowait' does not affect the taskwait for the 'depend' clause: its last
// argument (the nowait flag) is 0.
// CHECK-LABEL: define {{.*}}void @_Z13depend_nowaiti(
// CHECK: call void @__kmpc_omp_taskwait_deps_51(ptr @{{.+}}, i32 %{{.+}}, i32 1, ptr %{{.+}}, i32 0, ptr null, i32 0)
// CHECK: call void @_Z6f_dispi(
void depend_nowait(int a) {
#pragma omp dispatch depend(out:x) nowait
  f(a);
}

// A condition is evaluated even when there is nothing for it to select.
int sidefx();
void g(int);
// CHECK-LABEL: define {{.*}}void @_Z5noseli(
// CHECK: call {{.*}}i32 @_Z6sidefxv()
// CHECK-NOT: br i1
// CHECK: call void @_Z1gi(
void nosel(int a) {
#pragma omp dispatch novariants(sidefx())
  g(a);
}

// An assignment through an overloaded 'operator='. Each alternative assigns
// the result of its own call.
struct R {
  R &operator=(const R &);
  int v;
};
R r_disp(int);
R r_noctx(int);
#pragma omp declare variant(r_disp) match(construct={dispatch})
#pragma omp declare variant(r_noctx) match(user={condition(1)})
R r(int);

// CHECK-LABEL: define {{.*}}void @_Z11classassignR1Ribb(
// CHECK: br i1
// CHECK: call i32 @_Z1ri(
// CHECK: call {{.*}}@_ZN1RaSERKS_(
// CHECK: br i1
// CHECK: call i32 @_Z7r_noctxi(
// CHECK: call {{.*}}@_ZN1RaSERKS_(
// CHECK: call i32 @_Z6r_dispi(
// CHECK: call {{.*}}@_ZN1RaSERKS_(
void classassign(R &out, int a, bool c1, bool c2) {
#pragma omp dispatch novariants(c1) nocontext(c2)
  out = r(a);
}

// A dispatch region nested in a parallel region: the alternatives refer to the
// variables captured by the parallel region.
// CHECK-LABEL: define {{.*}}void @_Z6nestedibb(
// CHECK: call {{.*}}@__kmpc_fork_call(
// CHECK-LABEL: define internal void @_Z6nestedibb.omp_outlined(
// CHECK: [[OUT:%.+]] = load ptr, ptr %out.addr
// CHECK: br i1
// CHECK: [[R:%.+]] = call {{.*}}i32 @_Z1hi(i32 {{.*}}%{{.+}})
// CHECK-NEXT: store i32 [[R]], ptr [[OUT]]
// CHECK: br i1
// CHECK: [[R:%.+]] = call {{.*}}i32 @_Z7h_noctxi(
// CHECK-NEXT: store i32 [[R]], ptr [[OUT]]
// CHECK: [[R:%.+]] = call {{.*}}i32 @_Z6h_dispi(
// CHECK-NEXT: store i32 [[R]], ptr [[OUT]]
int h_disp(int);
int h_noctx(int);
#pragma omp declare variant(h_disp) match(construct={dispatch})
#pragma omp declare variant(h_noctx) match(user={condition(1)})
int h(int);
void nested(int a, bool c1, bool c2) {
  int out = 0;
#pragma omp parallel firstprivate(a) shared(out)
  {
#pragma omp dispatch novariants(c1) nocontext(c2)
    out = h(a);
  }
}

// The alternatives are built per instantiation, with the instantiated
// argument conversions.
template <typename T> void run(T t, bool c) {
#pragma omp dispatch novariants(c)
  f(t);
}
template void run<int>(int, bool);
template void run<long>(long, bool);

// CHECK-LABEL: define {{.*}}void @_Z3runIiEvT_b(
// CHECK: br i1
// CHECK: call void @_Z1fi(i32 noundef %
// CHECK: call void @_Z6f_dispi(i32 noundef %
// CHECK-LABEL: define {{.*}}void @_Z3runIlEvT_b(
// CHECK: br i1
// CHECK: [[CONV:%.+]] = trunc i64 %{{.+}} to i32
// CHECK-NEXT: call void @_Z1fi(i32 noundef [[CONV]])
// CHECK: [[CONV:%.+]] = trunc i64 %{{.+}} to i32
// CHECK-NEXT: call void @_Z6f_dispi(i32 noundef [[CONV]])

// Member function variants of a class template, called through implicit
// 'this' and through a pointer. Every alternative is called on the same
// object as the base function.
template <typename T> struct CT {
  void m_disp(T);
  void m_noctx(T);
#pragma omp declare variant(m_disp) match(construct={dispatch})
#pragma omp declare variant(m_noctx) match(user={condition(1)})
  void m(T);

  void use(T t, bool c1, bool c2) {
#pragma omp dispatch novariants(c1) nocontext(c2)
    m(t);
  }
};
template struct CT<int>;

template <typename T> void viaptr(CT<T> *p, T t, bool c1, bool c2) {
#pragma omp dispatch novariants(c1) nocontext(c2)
  p->m(t);
}
template void viaptr<int>(CT<int> *, int, bool, bool);

// CHECK-LABEL: define {{.*}}void @_ZN2CTIiE3useEibb(
// CHECK: [[THIS:%.+]] = load ptr, ptr %this.addr
// CHECK: br i1
// CHECK: call void @_ZN2CTIiE1mEi(ptr {{.*}}[[THIS]], i32
// CHECK: br i1
// CHECK: call void @_ZN2CTIiE7m_noctxEi(ptr {{.*}}[[THIS]], i32
// CHECK: call void @_ZN2CTIiE6m_dispEi(ptr {{.*}}[[THIS]], i32

// CHECK-LABEL: define {{.*}}void @_Z6viaptrIiEvP2CTIT_ES1_bb(
// CHECK: br i1
// CHECK: [[P:%.+]] = load ptr, ptr %p.addr
// CHECK: call void @_ZN2CTIiE1mEi(ptr {{.*}}[[P]], i32
// CHECK: br i1
// CHECK: [[P:%.+]] = load ptr, ptr %p.addr
// CHECK: call void @_ZN2CTIiE7m_noctxEi(ptr {{.*}}[[P]], i32
// CHECK: [[P:%.+]] = load ptr, ptr %p.addr
// CHECK: call void @_ZN2CTIiE6m_dispEi(ptr {{.*}}[[P]], i32

#endif
