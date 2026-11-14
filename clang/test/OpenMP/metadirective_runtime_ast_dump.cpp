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

// A dependent condition is resolved when the template is instantiated.
// CHECK-LABEL: FunctionTemplateDecl {{.*}} tmpl
// CHECK:      FunctionDecl {{.*}} tmpl 'void (T)'
// CHECK:      OMPMetaDirective
// CHECK:      FunctionDecl {{.*}} used tmpl 'void (int)' implicit_instantiation
// CHECK:      OMPMetaDirective
// CHECK-NEXT: OMPParallelDirective
// CHECK:      OMPSingleDirective
// CHECK:      DeclRefExpr {{.*}} 'flag' 'int'
template <typename T> void tmpl(T flag) {
#pragma omp metadirective when(user={condition(flag)}: parallel) otherwise(single)
  {
  }
}
void inst() { tmpl(1); }

// A block 'ordered' variant uses the statement, so 'otherwise' gets a copy.
// CHECK-LABEL: FunctionDecl {{.*}} ordered_block
// CHECK:      OMPMetaDirective
// CHECK-NEXT: OMPOrderedBlockAssocDirective
// CHECK-NOT:  NullStmt
// CHECK-COUNT-2: BinaryOperator {{.*}} 'int' lvalue '='
// CHECK-NOT:  NullStmt
// CHECK:      ImplicitCastExpr {{.*}} 'bool' <IntegralToBoolean>
void ordered_block(int flag, int n) {
#pragma omp for ordered
  for (int i = 0; i < n; ++i) {
#pragma omp metadirective when(user={condition(flag)}: ordered) otherwise(nothing)
    n = i;
  }
}

// An 'ordered' variant with 'doacross' is standalone, so the statement stays
// outside the variants.
// CHECK-LABEL: FunctionDecl {{.*}} ordered_doacross
// CHECK:      OMPMetaDirective
// CHECK-NEXT: OMPOrderedStandaloneDirective
// CHECK-NEXT: OMPDoacrossClause
// CHECK-NEXT: NullStmt
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'bool' <IntegralToBoolean>
// CHECK:      <<<NULL>>>
// CHECK-NEXT: BinaryOperator {{.*}} 'int' lvalue '='
void ordered_doacross(int flag, int n) {
#pragma omp for ordered(1)
  for (int i = 0; i < n; ++i) {
#pragma omp metadirective when(user={condition(flag)}: ordered doacross(source:)) otherwise(nothing)
    n = i;
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
