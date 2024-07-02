/* RUN: %clang_cc1 -ast-dump %s | FileCheck %s
 */

/* WG14 DR290: no
 * FLT_EVAL_METHOD and extra precision and/or range
 *
 * We retain an implicit conversion based on the float eval method being used
 * instead of dropping it due to the explicit cast. See GH86304 and C23 6.5.5p7.
 */

#pragma clang fp eval_method(double)
_Static_assert((float)(123.0F * 2.0F) == (float)246.0F, "");

// CHECK: StaticAssertDecl
// CHECK-NEXT: ImplicitCastExpr {{.*}} '_Bool' <IntegralToBoolean>
// CHECK-NEXT: BinaryOperator {{.*}} 'int' '=='
// NB: the following implicit cast is incorrect.
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'double' <FloatingCast> FPEvalMethod=1
// CHECK-NEXT: CStyleCastExpr {{.*}} 'float' <FloatingCast> FPEvalMethod=1

