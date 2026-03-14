// RUN: %clang_cc1 %s -Wno-unused-value -verify -fsyntax-only
// RUN: %clang_cc1 %s -Wno-unused-value -ast-dump -DAST_CHECK | FileCheck %s

struct S {};
void usage(float f, int i, double d) {
#ifdef AST_CHECK
  __builtin_fpclassify(d, 1, i, i, 3, d);
  //CHECK: CallExpr
  //CHECK-NEXT: ImplicitCastExpr
  //CHECK-SAME: <BuiltinFnToFnPtr>
  //CHECK-NEXT: DeclRefExpr
  //CHECK-SAME: '__builtin_fpclassify'
  //CHECK-NEXT: ImplicitCastExpr
  //CHECK-SAME: 'int' <FloatingToIntegral>
  //CHECK-NEXT: ImplicitCastExpr
  //CHECK-SAME: 'double' <LValueToRValue>
  //CHECK-NEXT: DeclRefExpr
  //CHECK-SAME: 'd' 'double'
  //CHECK-NEXT: IntegerLiteral
  //CHECK-NEXT: ImplicitCastExpr
  //CHECK-SAME: 'int' <LValueToRValue>
  //CHECK-NEXT: DeclRefExpr
  //CHECK-SAME: 'i' 'int'
  //CHECK-NEXT: ImplicitCastExpr
  //CHECK-SAME: 'int' <LValueToRValue>
  //CHECK-NEXT: DeclRefExpr
  //CHECK-SAME: 'i' 'int'
  //CHECK-NEXT: IntegerLiteral
  //CHECK-NEXT: ImplicitCastExpr
  //CHECK-SAME: 'double' <LValueToRValue>
  //CHECK-NEXT: DeclRefExpr
  //CHECK-SAME: 'd' 'double'

  __builtin_fpclassify(f, 1, i, i, 3, f);
  //CHECK: CallExpr
  //CHECK-NEXT: ImplicitCastExpr
  //CHECK-SAME: <BuiltinFnToFnPtr>
  //CHECK-NEXT: DeclRefExpr
  //CHECK-SAME: '__builtin_fpclassify'
  //CHECK-NEXT: ImplicitCastExpr
  //CHECK-SAME: 'int' <FloatingToIntegral>
  //CHECK-NEXT: ImplicitCastExpr
  //CHECK-SAME: 'float' <LValueToRValue>
  //CHECK-NEXT: DeclRefExpr
  //CHECK-SAME: 'f' 'float'
  //CHECK-NEXT: IntegerLiteral
  //CHECK-NEXT: ImplicitCastExpr
  //CHECK-SAME: 'int' <LValueToRValue>
  //CHECK-NEXT: DeclRefExpr
  //CHECK-SAME: 'i' 'int'
  //CHECK-NEXT: ImplicitCastExpr
  //CHECK-SAME: 'int' <LValueToRValue>
  //CHECK-NEXT: DeclRefExpr
  //CHECK-SAME: 'i' 'int'
  //CHECK-NEXT: IntegerLiteral
  //CHECK-NEXT: ImplicitCastExpr
  //CHECK-SAME: 'float' <LValueToRValue>
  //CHECK-NEXT: DeclRefExpr
  //CHECK-SAME: 'f' 'float'

  __builtin_isfinite(f);
  //CHECK: CallExpr
  //CHECK-NEXT: ImplicitCastExpr
  //CHECK-SAME: <BuiltinFnToFnPtr>
  //CHECK-NEXT: DeclRefExpr
  //CHECK-SAME: '__builtin_isfinite'
  //CHECK-NEXT: ImplicitCastExpr
  //CHECK-SAME: 'float' <LValueToRValue>
  //CHECK-NEXT: DeclRefExpr
  //CHECK-SAME: 'f' 'float'

  __builtin_isfinite(d);
  //CHECK: CallExpr
  //CHECK-NEXT: ImplicitCastExpr
  //CHECK-SAME: <BuiltinFnToFnPtr>
  //CHECK-NEXT: DeclRefExpr
  //CHECK-SAME: '__builtin_isfinite'
  //CHECK-NEXT: ImplicitCastExpr
  //CHECK-SAME: 'double' <LValueToRValue>
  //CHECK-NEXT: DeclRefExpr
  //CHECK-SAME: 'd' 'double'
#else
  struct S s;
  // expected-error@+1{{passing 'struct S' to parameter of incompatible type 'int'}}
  __builtin_fpclassify(d, s, i, i, 3, d);
  // expected-error@+1{{floating point classification requires argument of floating point type (passed in 'int')}}
  __builtin_fpclassify(d, 1, i, i, 3, i);
  // expected-error@+1{{floating point classification requires argument of floating point type (passed in 'int')}}
  __builtin_isfinite(i);
#endif
}
