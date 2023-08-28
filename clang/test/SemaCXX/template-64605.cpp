// RUN: %clang_cc1 -ast-dump -ast-dump-filter=b_64605 %s | FileCheck %s

// https://github.com/llvm/llvm-project/issues/64605

#pragma STDC FENV_ACCESS ON
template <typename>
int b_64605() {
  int x;
  if ((float)0xFFFFFFFF != (float)0x100000000) {
    x = 1;
  }
  return x;
}
int f() { return b_64605<void>(); }

// CHECK:      ImplicitCastExpr {{.*}} 'float' <IntegralToFloating> RoundingMath=1 AllowFEnvAccess=1
// CHECK-NEXT: IntegerLiteral {{.*}} 4294967295

// CHECK:      FunctionDecl {{.*}} b_64605 'int ()' implicit_instantiation
// CHECK-NEXT: TemplateArgument type 'void'

// CHECK:      ImplicitCastExpr {{.*}} 'float' <IntegralToFloating> RoundingMath=1 AllowFEnvAccess=1
// CHECK-NEXT: IntegerLiteral {{.*}} 4294967295
