// RUN: %clang_cc1 -ast-dump -std=c99 %s | FileCheck %s

void variadic(int i, ...);

void func(void) {
  // CHECK: FunctionDecl {{.*}} func 'void (void)'

  // Show that we correctly convert between two complex domains.
  _Complex float cf = 1.0f;
  _Complex double cd;

  cd = cf;
  // CHECK: BinaryOperator {{.*}} '_Complex double' '='
  // CHECK-NEXT: DeclRefExpr {{.*}} 'cd'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} '_Complex double' <FloatingComplexCast>
  // CHECK-NEXT: ImplicitCastExpr {{.*}} '_Complex float' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'cf'

  cf = cd;
  // CHECK: BinaryOperator {{.*}} '_Complex float' '='
  // CHECK-NEXT: DeclRefExpr {{.*}} 'cf'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} '_Complex float' <FloatingComplexCast>
  // CHECK-NEXT: ImplicitCastExpr {{.*}} '_Complex double' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'cd'

  // Show that we correctly convert to the common type of a complex and real.
  // This should convert the _Complex float to a _Complex double ("without
  // change of domain" c.f. C99 6.3.1.8p1).
  (void)(cf + 1.0);
  // CHECK: BinaryOperator {{.*}} '_Complex double' '+'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} '_Complex double' <FloatingComplexCast>
  // CHECK-NEXT: ImplicitCastExpr {{.*}} '_Complex float' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'cf'
  // CHECK-NEXT: FloatingLiteral {{.*}} 'double' 1.0

  // This should convert the float constant to double, then produce a
  // _Complex double.
  (void)(cd + 1.0f);
  // CHECK: BinaryOperator {{.*}} '_Complex double' '+'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} '_Complex double' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'cd'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'double' <FloatingCast>
  // CHECK-NEXT: FloatingLiteral {{.*}} 'float' 1.0

  // This should convert the int constant to float, then produce a
  // _Complex float.
  (void)(cf + 1);
  // CHECK: BinaryOperator {{.*}} '_Complex float' '+'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} '_Complex float' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'cf'
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
  // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1

  // Show that we do not promote a _Complex float to _Complex double as part of
  // the default argument promotions when passing to a variadic function.
  variadic(1, cf);
  // CHECK: CallExpr
  // CHECK-NEXT: ImplicitCastExpr {{.*}} <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'variadic'
  // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
  // CHECK-NEXT: ImplicitCastExpr {{.*}} '_Complex float' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'cf'
}

