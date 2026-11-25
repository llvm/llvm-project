// RUN: %clang_cc1 -verify -Wno-vla -fopenmp -x c++ -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-apple-darwin10 -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -verify -Wno-vla -fopenmp-simd -x c++ -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-apple-darwin10 -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// Task reduction of an array with a declare reduction that has no initializer
// clause: the initializer function must not read the original item.

struct A {
  int a;
};
#pragma omp declare reduction(+ : A : omp_out.a += omp_in.a)

// Reproducer from GH141228.
void test_for() {
  A a, b[2];
#pragma omp parallel
  {
#pragma omp for reduction(task, + : a, b)
    for (int i = 0; i < 5; i++)
      ;
  }
}

// CHECK-LABEL: define internal void @_Z8test_forv.omp_outlined(
// CHECK: store ptr @[[A_INIT:.red_init.[.0-9]*]], ptr
// CHECK: store ptr @[[B_INIT:.red_init.[.0-9]*]], ptr
// CHECK: call ptr @__kmpc_taskred_modifier_init(
// CHECK: define internal void @[[A_INIT]](ptr noalias noundef %{{.+}}, ptr noalias noundef %{{.+}})
// CHECK-NOT: omp.arrayinit
// CHECK: ret void
// CHECK: define internal void @[[B_INIT]](ptr noalias noundef %{{.+}}, ptr noalias noundef %{{.+}})
// CHECK-NOT: srcElementPast
// CHECK: omp.arrayinit.body:
// CHECK-NEXT: [[DST:%.+]] = phi ptr [
// CHECK-NEXT: getelementptr %struct.A, ptr [[DST]], i32 1
// CHECK: ret void

void test_sections() {
  A b[2];
#pragma omp parallel
  {
#pragma omp sections reduction(task, + : b)
    {
      b[0].a = 1;
    }
  }
}

// CHECK-LABEL: define internal void @_Z13test_sectionsv.omp_outlined(
// CHECK: store ptr @[[SB_INIT:.red_init.[.0-9]*]], ptr
// CHECK: call ptr @__kmpc_taskred_modifier_init(
// CHECK: define internal void @[[SB_INIT]](ptr noalias noundef %{{.+}}, ptr noalias noundef %{{.+}})
// CHECK-NOT: srcElementPast
// CHECK: omp.arrayinit.body:
// CHECK-NEXT: [[DST:%.+]] = phi ptr [
// CHECK-NEXT: getelementptr %struct.A, ptr [[DST]], i32 1
// CHECK: ret void

void test_taskgroup() {
  A m[2][3];
#pragma omp taskgroup task_reduction(+ : m)
  {
#pragma omp task in_reduction(+ : m)
    m[1][2].a = 1;
  }
}

// CHECK-LABEL: define {{.*}}void @_Z14test_taskgroupv(
// CHECK: store ptr @[[M_INIT:.red_init.[.0-9]*]], ptr
// CHECK: call ptr @__kmpc_taskred_init(
// CHECK: define internal void @[[M_INIT]](ptr noalias noundef %{{.+}}, ptr noalias noundef %{{.+}})
// CHECK-NOT: srcElementPast
// CHECK: omp.arrayinit.body:
// CHECK-NEXT: [[DST:%.+]] = phi ptr [
// CHECK-NEXT: getelementptr %struct.A, ptr [[DST]], i32 1
// CHECK: ret void

void test_vla(int n) {
  A v[n];
#pragma omp parallel
  {
#pragma omp for reduction(task, + : v)
    for (int i = 0; i < 5; i++)
      ;
  }
}

// CHECK-LABEL: define internal void @_Z8test_vlai.omp_outlined(
// CHECK: store ptr @[[V_INIT:.red_init.[.0-9]*]], ptr
// CHECK: call ptr @__kmpc_taskred_modifier_init(
// CHECK: define internal void @[[V_INIT]](ptr noalias noundef %{{.+}}, ptr noalias noundef %{{.+}})
// CHECK-NOT: srcElementPast
// CHECK: omp.arrayinit.body:
// CHECK-NEXT: [[DST:%.+]] = phi ptr [
// CHECK-NEXT: getelementptr %struct.A, ptr [[DST]], i32 1
// CHECK: ret void

// No default-initialization expression: elements are zero-initialized.
#pragma omp declare reduction(my_add : int : omp_out += omp_in)
void test_int() {
  int c[2];
#pragma omp parallel
  {
#pragma omp for reduction(task, my_add : c)
    for (int i = 0; i < 5; i++)
      ;
  }
}

// CHECK-LABEL: define internal void @_Z8test_intv.omp_outlined(
// CHECK: store ptr @[[C_INIT:.red_init.[.0-9]*]], ptr
// CHECK: call ptr @__kmpc_taskred_modifier_init(
// CHECK: define internal void @[[C_INIT]](ptr noalias noundef %{{.+}}, ptr noalias noundef %{{.+}})
// CHECK-NOT: srcElementPast
// CHECK: omp.arrayinit.body:
// CHECK-NEXT: [[DST:%.+]] = phi ptr [
// CHECK-NEXT: [[ZERO:%.+]] = load i32, ptr @.init
// CHECK-NEXT: store i32 [[ZERO]], ptr [[DST]]
// CHECK-NEXT: getelementptr i32, ptr [[DST]], i32 1
// CHECK: ret void

// Non-trivial element type.
struct S {
  int a;
  S() : a(0) {}
  ~S() {}
};
#pragma omp declare reduction(+ : S : omp_out.a += omp_in.a)
void test_nontrivial() {
  S s[2];
#pragma omp parallel
  {
#pragma omp for reduction(task, + : s)
    for (int i = 0; i < 5; i++)
      ;
  }
}

// CHECK-LABEL: define internal void @_Z15test_nontrivialv.omp_outlined(
// CHECK: store ptr @[[S_INIT:.red_init.[.0-9]*]], ptr
// CHECK: store ptr @[[S_FINI:.red_fini.[.0-9]*]], ptr
// CHECK: call ptr @__kmpc_taskred_modifier_init(
// CHECK: define internal void @[[S_INIT]](ptr noalias noundef %{{.+}}, ptr noalias noundef %{{.+}})
// CHECK-NOT: srcElementPast
// CHECK: omp.arrayinit.body:
// CHECK-NEXT: [[DST:%.+]] = phi ptr [
// CHECK-NEXT: call void @_ZN1SC1Ev(ptr {{[^,]*}}[[DST]])
// CHECK-NEXT: getelementptr %struct.S, ptr [[DST]], i32 1
// CHECK: ret void
// CHECK: define internal void @[[S_FINI]](ptr noundef %{{.+}})
// CHECK: call void @_ZN1SD1Ev(
// CHECK: ret void

// Control: the initializer reads the original item.
struct B {
  int a;
};
#pragma omp declare reduction(+ : B : omp_out.a += omp_in.a) initializer(omp_priv = omp_orig)
void test_with_initializer() {
  B b[2];
#pragma omp parallel
  {
#pragma omp for reduction(task, + : b)
    for (int i = 0; i < 5; i++)
      ;
  }
}

// CHECK-LABEL: define internal void @_Z21test_with_initializerv.omp_outlined(
// CHECK: store ptr @[[WB_INIT:.red_init.[.0-9]*]], ptr
// CHECK: call ptr @__kmpc_taskred_modifier_init(
// CHECK: define internal void @[[WB_INIT]](ptr noalias noundef %{{.+}}, ptr noalias noundef %{{.+}})
// CHECK: omp.arrayinit.body:
// CHECK-NEXT: [[SRC:%.+]] = phi ptr [
// CHECK-NEXT: [[DST:%.+]] = phi ptr [
// CHECK-NEXT: call void @.omp_initializer.{{[.0-9]*}}(ptr {{[^,]*}}[[DST]], ptr {{[^,]*}}[[SRC]])
// CHECK-NEXT: getelementptr %struct.B, ptr [[SRC]], i32 1
// CHECK: ret void

// Control: non-task reduction.
void test_nontask() {
  A b[2];
#pragma omp parallel for reduction(+ : b)
  for (int i = 0; i < 5; i++)
    ;
}

// CHECK-LABEL: define internal void @_Z12test_nontaskv.omp_outlined(
// CHECK: omp.arrayinit.body:
// CHECK-NEXT: [[DST:%.+]] = phi ptr [
// CHECK-NEXT: getelementptr %struct.A, ptr [[DST]], i32 1

#endif
