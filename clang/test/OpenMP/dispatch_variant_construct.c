// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=60 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -ast-dump %s | FileCheck %s

// expected-no-diagnostics

// The call under dispatch can be replaced by a variant that matches the
// dispatch construct.

void f_par(void) {}
void f_par_dispatch(void) {}

#pragma omp declare variant(f_par) match(construct = {parallel})
#pragma omp declare variant(f_par_dispatch)                                   \
    match(construct = {parallel, dispatch})
void f(void) {}

void g_dispatch(void) {}

#pragma omp declare variant(g_dispatch) match(construct = {dispatch})
void g(void) {}

// CHECK-LABEL: FunctionDecl {{.*}} test_parallel
// CHECK:       OMPDispatchDirective
// CHECK:       PseudoObjectExpr
// CHECK:       DeclRefExpr {{.*}} 'f' 'void (void)'
// CHECK:       DeclRefExpr {{.*}} 'f_par_dispatch' 'void (void)'
void test_parallel(void) {
#pragma omp parallel
  {
#pragma omp dispatch
    f();
  }
}

// CHECK-LABEL: FunctionDecl {{.*}} test_top_level
// CHECK:       OMPDispatchDirective
// CHECK:       PseudoObjectExpr
// CHECK:       DeclRefExpr {{.*}} 'g' 'void (void)'
// CHECK:       DeclRefExpr {{.*}} 'g_dispatch' 'void (void)'
void test_top_level(void) {
#pragma omp dispatch
  g();
}
