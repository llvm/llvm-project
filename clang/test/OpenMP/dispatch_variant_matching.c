// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 -verify -ast-dump %s \
// RUN:   | FileCheck %s --implicit-check-not=PseudoObjectExpr
// RUN: %clang_cc1 -x c++ -fopenmp -fopenmp-version=52 -verify -ast-dump %s \
// RUN:   | FileCheck %s --implicit-check-not=PseudoObjectExpr
// expected-no-diagnostics

// Clang omits the implementation-defined DISPATCH construct trait. In
// particular, it must not select dispatch variants in argument expressions.
int g_variant(void);
#pragma omp declare variant(g_variant) match(construct = {dispatch})
int g(void);

int f_variant(int);
#pragma omp declare variant(f_variant) match(construct = {dispatch})
int f(int);
void plain(int);

// CHECK-LABEL: FunctionDecl {{.*}} test_argument
// CHECK: OMPDispatchDirective
// CHECK: DeclRefExpr {{.*}} Function {{.*}} 'plain'
// CHECK: DeclRefExpr {{.*}} Function {{.*}} 'g'
void test_argument(void) {
#pragma omp dispatch
  plain(g());
}

// CHECK-LABEL: FunctionDecl {{.*}} test_target
// CHECK: OMPDispatchDirective
// CHECK: DeclRefExpr {{.*}} Function {{.*}} 'f'
// CHECK: DeclRefExpr {{.*}} Function {{.*}} 'g'
void test_target(void) {
#pragma omp dispatch
  f(g());
}

// CHECK-LABEL: FunctionDecl {{.*}} test_assignment
// CHECK: OMPDispatchDirective
// CHECK: DeclRefExpr {{.*}} Function {{.*}} 'f'
// CHECK: DeclRefExpr {{.*}} Function {{.*}} 'g'
void test_assignment(int *result) {
#pragma omp dispatch
  *result = f(g());
}
