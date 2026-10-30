// RUN: %clang_cc1 -fopenmp -fopenmp-version=52 -std=c++11 \
// RUN:   -fopenmp-metadirective-runtime-selection -ast-dump %s | FileCheck %s

// CHECK-LABEL: FunctionDecl {{.*}} when_otherwise
// CHECK:      OMPMetaDirective
// CHECK-NEXT: OMPParallelDirective
// CHECK:      OMPSingleDirective
// CHECK:      ImplicitCastExpr {{.*}} 'bool' <IntegralToBoolean>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'flag' 'int'
// CHECK-NEXT: <<<NULL>>>
void when_otherwise(int flag) {
#pragma omp metadirective when(user={condition(flag)}: parallel) otherwise(single)
  {
  }
}

// Without 'otherwise', the last variant is the plain associated statement.
// CHECK-LABEL: FunctionDecl {{.*}} no_otherwise
// CHECK:      OMPMetaDirective
// CHECK-NEXT: OMPParallelDirective
// CHECK:      {{^ *}}|-CompoundStmt
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'bool' <IntegralToBoolean>
// CHECK:      <<<NULL>>>
void no_otherwise(int flag) {
#pragma omp metadirective when(user={condition(flag)}: parallel)
  {
  }
}
