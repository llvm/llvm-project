// RUN: %clang_cc1 -fsyntax-only -triple x86_64-pc-linux -std=c++11 -fcxx-exceptions -fdelayed-template-parsing -ast-dump %s \
// RUN: | FileCheck %s

#pragma STDC FENV_ROUND FE_DOWNWARD
#pragma float_control(precise, off)

template <typename T>
__attribute__((optnone)) T func_22(T x, T y) {
  return x + y;
}

// CHECK-LABEL: FunctionTemplateDecl {{.*}} func_22
// CHECK:         FunctionDecl {{.*}} func_22 'T (T, T)'
// CHECK:           CompoundStmt {{.*}} ConstRoundingMode=downward MathErrno=1
// CHECK:             ReturnStmt
// CHECK:               BinaryOperator {{.*}} '+' ConstRoundingMode=downward MathErrno=1
// CHECK:         FunctionDecl {{.*}} func_22 'float (float, float)'
// CHECK:           CompoundStmt {{.*}} ConstRoundingMode=downward MathErrno=1
// CHECK:             ReturnStmt
// CHECK:               BinaryOperator {{.*}} 'float' '+' ConstRoundingMode=downward MathErrno=1

float func_23(float x, float y) {
  return func_22(x, y);
}
