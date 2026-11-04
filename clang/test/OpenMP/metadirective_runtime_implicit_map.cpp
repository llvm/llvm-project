// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=52 \
// RUN:   -fopenmp-metadirective-runtime-selection \
// RUN:   -fopenmp-targets=x86_64-pc-linux-gnu -ast-dump %s | FileCheck %s
// expected-no-diagnostics

// A member used only inside a nested runtime metadirective is implicitly
// mapped by the enclosing target region.
struct S {
  int a;
  // CHECK-LABEL: CXXMethodDecl {{.*}} f_meta
  // CHECK:       OMPTargetDirective
  // CHECK:       OMPMapClause {{.*}} <implicit>
  // CHECK-NEXT:  MemberExpr {{.*}} ->a
  void f_meta(int flag) {
#pragma omp target
    {
#pragma omp metadirective when(user={condition(flag)}: parallel) otherwise(single)
      a = 1;
    }
  }
};

void use(S &s) { s.f_meta(1); }
