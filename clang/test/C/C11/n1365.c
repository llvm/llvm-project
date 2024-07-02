// RUN: %clang_cc1 -ast-dump %s | FileCheck %s

/* WG14 N1365: Clang 16
 * Constant expressions
 */

// Note: we don't allow you to expand __FLT_EVAL_METHOD__ in the presence of a
// pragma that changes its value. However, we can test that we have the correct
// constant expression behavior by testing that the AST has the correct implicit
// casts, which also specify that the cast was inserted due to an evaluation
// method requirement.
void func(void) {
  {
    #pragma clang fp eval_method(double)
    _Static_assert(123.0F * 2.0F == 246.0F, "");
    // CHECK: StaticAssertDecl
    // CHECK-NEXT: ImplicitCastExpr {{.*}} '_Bool' <IntegralToBoolean>
    // CHECK-NEXT: BinaryOperator {{.*}} 'int' '=='
    // CHECK-NEXT: BinaryOperator {{.*}} 'double' '*' FPEvalMethod=1
    // CHECK-NEXT: ImplicitCastExpr {{.*}} 'double' <FloatingCast> FPEvalMethod=1
    // CHECK-NEXT: FloatingLiteral
    // CHECK-NEXT: ImplicitCastExpr {{.*}} 'double' <FloatingCast> FPEvalMethod=1
    // CHECK-NEXT: FloatingLiteral

    // Ensure that a cast removes the extra precision.
    _Static_assert(123.0F * 2.0F == 246.0F, "");
    // CHECK: StaticAssertDecl
    // CHECK-NEXT: ImplicitCastExpr {{.*}} '_Bool' <IntegralToBoolean>
    // CHECK-NEXT: BinaryOperator {{.*}} 'int' '=='
    // CHECK-NEXT: BinaryOperator {{.*}} 'double' '*' FPEvalMethod=1
    // CHECK-NEXT: ImplicitCastExpr {{.*}} 'double' <FloatingCast> FPEvalMethod=1
    // CHECK-NEXT: FloatingLiteral
    // CHECK-NEXT: ImplicitCastExpr {{.*}} 'double' <FloatingCast> FPEvalMethod=1
    // CHECK-NEXT: FloatingLiteral
  }

  {
    #pragma clang fp eval_method(extended)
    _Static_assert(123.0F * 2.0F == 246.0F, "");
    // CHECK: StaticAssertDecl
    // CHECK-NEXT: ImplicitCastExpr {{.*}} '_Bool' <IntegralToBoolean>
    // CHECK-NEXT: BinaryOperator {{.*}} 'int' '=='
    // CHECK-NEXT: BinaryOperator {{.*}} 'long double' '*' FPEvalMethod=2
    // CHECK-NEXT: ImplicitCastExpr {{.*}} 'long double' <FloatingCast> FPEvalMethod=2
    // CHECK-NEXT: FloatingLiteral
    // CHECK-NEXT: ImplicitCastExpr {{.*}} 'long double' <FloatingCast> FPEvalMethod=2
    // CHECK-NEXT: FloatingLiteral
  }

  {
    #pragma clang fp eval_method(source)
    _Static_assert(123.0F * 2.0F == 246.0F, "");
    // CHECK: StaticAssertDecl
    // CHECK-NEXT: ImplicitCastExpr {{.*}} '_Bool' <IntegralToBoolean>
    // CHECK-NEXT: BinaryOperator {{.*}} 'int' '=='
    // CHECK-NEXT: BinaryOperator {{.*}} 'float' '*' FPEvalMethod=0
    // CHECK-NEXT: FloatingLiteral
    // CHECK-NEXT: FloatingLiteral
  }
}
