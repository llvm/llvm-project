// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

// Cases adapted from clang/test/OpenMP/critical_codegen.cpp, without
// exceptions. Language-independent cases (names, hints, nesting, combinations
// with other directives) are in critical.c.

void foo();

int critical_in_main() {
  char a;
#pragma omp critical
  a = 2;
#pragma omp critical(the_name)
  foo();
#pragma omp critical(the_name1) hint(4)
  foo();
  if (a)
#pragma omp critical(the_name)
    while (1)
      ;
  foo();
  return a;
}

void lambda_critical(int a, int b) {
  auto l = [=]() {
#pragma omp critical
    {
      int c = a + b;
    }
  };
  l();
  auto l1 = [=]() {
#pragma omp parallel
#pragma omp critical
    {
      int c = a + b;
    }
  };
  l1();
}

struct S {
  int a;
};

void critical_ref(S &s) {
  ++s.a;
#pragma omp critical
  ++s.a;
}

// Every instantiation uses the same lock.
template <typename T> T critical_template(T x) {
#pragma omp critical(tmpl)
  x += 1;
  return x;
}

int use_template() { return critical_template(1) + (int)critical_template(2.0); }

// CIR-DAG: omp.critical.declare @the_name{{ }}
// CIR-DAG: omp.critical.declare @the_name1 hint(nonspeculative)
// CIR-DAG: omp.critical.declare @tmpl

// CIR-LABEL: cir.func {{.*}}@_Z16critical_in_mainv()
// CIR:         %[[A:.*]] = cir.alloca "a"
// CIR:         omp.critical {
// CIR:           cir.store {{.*}}, %[[A]]
// CIR:           omp.terminator
// CIR:         }
// CIR:         omp.critical(@the_name) {
// CIR:           cir.call @_Z3foov()
// CIR:           omp.terminator
// CIR:         }
// CIR:         omp.critical(@the_name1) {
// CIR:           cir.call @_Z3foov()
// CIR:           omp.terminator
// CIR:         }
// CIR:         cir.if
// CIR:           omp.critical(@the_name) {
// CIR:             cir.while {
// CIR:             omp.terminator
// CIR:           }
// CIR:         cir.call @_Z3foov()

// CIR-LABEL: cir.func {{.*}}@_ZZ15lambda_criticaliiENK3$_0clEv(
// CIR:         omp.critical {
// CIR:           cir.alloca "c"
// CIR:           omp.terminator
// CIR:         }

// CIR-LABEL: cir.func {{.*}}@_ZZ15lambda_criticaliiENK3$_1clEv(
// CIR:         omp.parallel {
// CIR:           omp.critical {
// CIR:             cir.alloca "c"
// CIR:             omp.terminator
// CIR:           }
// CIR:           omp.terminator
// CIR:         }

// CIR-LABEL: cir.func {{.*}}@_Z12critical_refR1S(
// CIR:         cir.inc
// CIR:         omp.critical {
// CIR:           cir.get_member {{.*}}{name = "a"}
// CIR:           cir.inc
// CIR:           omp.terminator
// CIR:         }

// CIR-LABEL: cir.func {{.*}}@_Z17critical_templateIiET_S0_(
// CIR:         omp.critical(@tmpl) {
// CIR:           omp.terminator
// CIR:         }

// CIR-LABEL: cir.func {{.*}}@_Z17critical_templateIdET_S0_(
// CIR:         omp.critical(@tmpl) {
// CIR:           omp.terminator
// CIR:         }

// LLVM-DAG: @[[UNNAMED:.gomp_critical_user_.var]] = {{.*}}global [8 x i32] zeroinitializer
// LLVM-DAG: @[[THE_NAME:.gomp_critical_user_the_name.var]] = {{.*}}global [8 x i32] zeroinitializer
// LLVM-DAG: @[[THE_NAME1:.gomp_critical_user_the_name1.var]] = {{.*}}global [8 x i32] zeroinitializer
// LLVM-DAG: @[[TMPL:.gomp_critical_user_tmpl.var]] = {{.*}}global [8 x i32] zeroinitializer

// LLVM-LABEL: define {{.*}}i32 @_Z16critical_in_mainv()
// LLVM:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])
// LLVM:         store i8 2,
// LLVM:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])
// LLVM:         call void @__kmpc_critical_with_hint({{.*}}, i32 {{.*}}, ptr @[[THE_NAME]], i32 0)
// LLVM:         call void @_Z3foov()
// LLVM:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[THE_NAME]])
// LLVM:         call void @__kmpc_critical_with_hint({{.*}}, i32 {{.*}}, ptr @[[THE_NAME1]], i32 4)
// LLVM:         call void @_Z3foov()
// LLVM:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[THE_NAME1]])
// LLVM:         call void @__kmpc_critical_with_hint({{.*}}, i32 {{.*}}, ptr @[[THE_NAME]], i32 0)
// LLVM:         call void @_Z3foov()

// LLVM-LABEL: define {{.*}}void @"_ZZ15lambda_criticaliiENK3$_0clEv"(
// LLVM:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])
// LLVM:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])

// LLVM-LABEL: define {{.*}}void @"_ZZ15lambda_criticaliiENK3$_1clEv"(
// LLVM:         call {{.*}}@__kmpc_fork_call(
// LLVM-LABEL: define {{.*}}void @"_ZZ15lambda_criticaliiENK3$_1clEv..omp_par"(
// LLVM:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])
// LLVM:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])

// LLVM-LABEL: define {{.*}}void @_Z12critical_refR1S(
// LLVM:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])
// LLVM:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])

// LLVM-LABEL: define {{.*}}i32 @_Z17critical_templateIiET_S0_(
// LLVM:         call void @__kmpc_critical_with_hint({{.*}}, i32 {{.*}}, ptr @[[TMPL]], i32 0)
// LLVM:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[TMPL]])

// LLVM-LABEL: define {{.*}}double @_Z17critical_templateIdET_S0_(
// LLVM:         call void @__kmpc_critical_with_hint({{.*}}, i32 {{.*}}, ptr @[[TMPL]], i32 0)
// LLVM:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[TMPL]])

// OGCG-DAG: @[[UNNAMED:.gomp_critical_user_.var]] = {{.*}}global [8 x i32] zeroinitializer
// OGCG-DAG: @[[THE_NAME:.gomp_critical_user_the_name.var]] = {{.*}}global [8 x i32] zeroinitializer
// OGCG-DAG: @[[THE_NAME1:.gomp_critical_user_the_name1.var]] = {{.*}}global [8 x i32] zeroinitializer
// OGCG-DAG: @[[TMPL:.gomp_critical_user_tmpl.var]] = {{.*}}global [8 x i32] zeroinitializer

// OGCG-LABEL: define {{.*}}i32 @_Z16critical_in_mainv()
// OGCG:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])
// OGCG:         store i8 2,
// OGCG:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])
// OGCG:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[THE_NAME]])
// OGCG:         call void @_Z3foov()
// OGCG:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[THE_NAME]])
// OGCG:         call void @__kmpc_critical_with_hint({{.*}}, i32 {{.*}}, ptr @[[THE_NAME1]], i32 4)
// OGCG:         call void @_Z3foov()
// OGCG:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[THE_NAME1]])
// OGCG:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[THE_NAME]])
// OGCG:         call void @_Z3foov()

// OGCG-LABEL: define {{.*}}void @"_ZZ15lambda_criticaliiENK3$_0clEv"(
// OGCG:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])
// OGCG:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])

// OGCG-LABEL: define {{.*}}void @"_ZZ15lambda_criticaliiENK3$_1clEv"(
// OGCG:         call {{.*}}@__kmpc_fork_call(
// OGCG-LABEL: define {{.*}}void @"_ZZ15lambda_criticaliiENK3$_1clEv.omp_outlined"(
// OGCG:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])
// OGCG:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])

// OGCG-LABEL: define {{.*}}void @_Z12critical_refR1S(
// OGCG:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])
// OGCG:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[UNNAMED]])

// OGCG-LABEL: define {{.*}}i32 @_Z17critical_templateIiET_S0_(
// OGCG:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[TMPL]])
// OGCG:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[TMPL]])

// OGCG-LABEL: define {{.*}}double @_Z17critical_templateIdET_S0_(
// OGCG:         call void @__kmpc_critical({{.*}}, i32 {{.*}}, ptr @[[TMPL]])
// OGCG:         call void @__kmpc_end_critical({{.*}}, i32 {{.*}}, ptr @[[TMPL]])
