// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -ast-dump %s | FileCheck %s
// expected-no-diagnostics

typedef __attribute__((ext_vector_type(2)))  char char2;

void vectorIncrementDecrementOps() {
  // CHECK: FunctionDecl {{.*}} <{{.*}}, line:{{.*}}> line:{{.*}} vectorIncrementDecrementOps 'void (void)'{{$}}
  // CHECK: CStyleCastExpr {{.*}} <col:{{.*}}> 'char2':'char __attribute__((ext_vector_type(2)))' <VectorSplat>{{$}}
  // CHECK-NEXT: ImplicitCastExpr {{.*}} <col:{{.*}}> 'char' <IntegralCast> part_of_explicit_cast{{$}}
  // CHECK-NEXT: IntegerLiteral {{.*}} <col:{{.*}}> 'int' 1{{$}}
  char2 A = (char2)(1);
  A++;
}
