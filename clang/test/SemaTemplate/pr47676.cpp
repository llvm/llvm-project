// RUN: %clang_cc1 -triple=powerpc64le-unknown-linux-gnu \
// RUN:            -target-feature +altivec -fsyntax-only -ast-dump \
// RUN:            -xc++ < %s \
// RUN:   | FileCheck %s

// Ensures that casts to AltiVec type with a dependent expression operand does
// not hit the assertion failure reported in PR47676. Further checks that casts
// to AltiVec type with a dependent expression operand is, on instantiation,
// able to correctly differentiate between a splat case and a bitcast case.
template <typename T> void f(T *tp) {
  extern void g(int, ...);
  g(0, (__vector int)(*tp));
  g(0, (__vector int)*tp);
}

void g(void) {
  f<__vector float>(nullptr);
//      CHECK: | |-FunctionDecl {{.*}} f 'void (__vector float *)'

//      CHECK: |   | `-CStyleCastExpr {{.*}} '__vector int' <NoOp>
// CHECK-NEXT: |   |   `-ImplicitCastExpr {{.*}} '__vector int' <BitCast>
// CHECK-NEXT: |   |     `-ImplicitCastExpr {{.*}}'__vector float' <LValueToRValue>

//      CHECK: |     `-CStyleCastExpr {{.*}} '__vector int' <NoOp>
// CHECK-NEXT: |       `-ImplicitCastExpr {{.*}} '__vector int' <BitCast>
// CHECK-NEXT: |         `-ImplicitCastExpr {{.*}}'__vector float' <LValueToRValue>

  f<double>(nullptr);
//      CHECK: | `-FunctionDecl {{.*}} f 'void (double *)'

//      CHECK: |     | `-CStyleCastExpr {{.*}} '__vector int' <VectorSplat>
// CHECK-NEXT: |     |   `-ImplicitCastExpr {{.*}} 'int' <FloatingToIntegral>
// CHECK-NEXT: |     |     `-ImplicitCastExpr {{.*}}'double' <LValueToRValue>

//      CHECK: |       `-CStyleCastExpr {{.*}} '__vector int' <VectorSplat>
// CHECK-NEXT: |         `-ImplicitCastExpr {{.*}} 'int' <FloatingToIntegral>
// CHECK-NEXT: |           `-ImplicitCastExpr {{.*}}'double' <LValueToRValue>
}
