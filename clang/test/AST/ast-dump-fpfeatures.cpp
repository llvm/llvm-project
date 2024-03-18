// Test without serialization:
// RUN: %clang_cc1 -fsyntax-only -triple x86_64-pc-linux -std=c++11 -ast-dump %s \
// RUN: | FileCheck --strict-whitespace %s

// Test with serialization:
// RUN: %clang_cc1 -triple x86_64-pc-linux -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -triple x86_64-pc-linux -include-pch %t -ast-dump-all /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck --strict-whitespace %s

float func_01(float x);

template <typename T>
T func_02(T x) {
#pragma STDC FP_CONTRACT ON
  return func_01(x);
}

float func_03(float x) {
#pragma STDC FP_CONTRACT OFF
  return func_02(x);
}

// CHECK:      FunctionTemplateDecl {{.*}} func_02
// CHECK:        FunctionDecl {{.*}} func_02 'float (float)'
// CHECK-NEXT:     TemplateArgument type 'float'
// CHECK-NEXT:       BuiltinType {{.*}} 'float'
// CHECK-NEXT:     ParmVarDecl {{.*}} x 'float'
// CHECK-NEXT:     CompoundStmt
// CHECK-NEXT:       ReturnStmt
// CHECK-NEXT:         CallExpr {{.*}} FPContractMode=1

// CHECK:      FunctionDecl {{.*}} func_03 'float (float)'
// CHECK-NEXT:   ParmVarDecl {{.*}} x 'float'
// CHECK-NEXT:     CompoundStmt
// CHECK-NEXT:       ReturnStmt
// CHECK-NEXT:         CallExpr {{.*}} FPContractMode=0

int func_04(float x) {
#pragma STDC FP_CONTRACT ON
  return x;
}

// CHECK:      FunctionDecl {{.*}} func_04 'int (float)'
// CHECK-NEXT:   ParmVarDecl {{.*}} x 'float'
// CHECK-NEXT:   CompoundStmt
// CHECK-NEXT:     ReturnStmt
// CHECK-NEXT:       ImplicitCastExpr {{.*}} 'int' <FloatingToIntegral> FPContractMode=1

float func_05(double x) {
#pragma STDC FP_CONTRACT ON
  return (float)x;
}

// CHECK:      FunctionDecl {{.*}} func_05 'float (double)'
// CHECK-NEXT:   ParmVarDecl {{.*}} x 'double'
// CHECK-NEXT:   CompoundStmt
// CHECK-NEXT:     ReturnStmt
// CHECK-NEXT:       CStyleCastExpr {{.*}} FPContractMode=1

float func_06(double x) {
#pragma STDC FP_CONTRACT ON
  return float(x);
}

// CHECK:      FunctionDecl {{.*}} func_06 'float (double)'
// CHECK-NEXT:   ParmVarDecl {{.*}} x 'double'
// CHECK-NEXT:   CompoundStmt
// CHECK-NEXT:     ReturnStmt
// CHECK-NEXT:       CXXFunctionalCastExpr {{.*}} FPContractMode=1

float func_07(double x) {
#pragma STDC FP_CONTRACT ON
  return static_cast<float>(x);
}

// CHECK:      FunctionDecl {{.*}} func_07 'float (double)'
// CHECK-NEXT:   ParmVarDecl {{.*}} x 'double'
// CHECK-NEXT:   CompoundStmt
// CHECK-NEXT:     ReturnStmt
// CHECK-NEXT:       CXXStaticCastExpr {{.*}} FPContractMode=1

#pragma STDC FENV_ROUND FE_DOWNWARD

float func_10(float x, float y) {
  return x + y;
}

// CHECK-LABEL: FunctionDecl {{.*}} func_10 'float (float, float)'
// CHECK:         BinaryOperator {{.*}} 'float' '+' ConstRoundingMode=downward

float func_11(float x, float y) {
  if (x < 0) {
    #pragma STDC FENV_ROUND FE_UPWARD
    return x + y;
  }
  return x - y;
}

// CHECK-LABEL: FunctionDecl {{.*}} func_11 'float (float, float)'
// CHECK:         BinaryOperator {{.*}} 'float' '+' ConstRoundingMode=upward
// CHECK:         BinaryOperator {{.*}} 'float' '-' ConstRoundingMode=downward


#pragma STDC FENV_ROUND FE_DYNAMIC

float func_12(float x, float y) {
  return x + y;
}

// CHECK-LABEL: FunctionDecl {{.*}} func_12 'float (float, float)'
// CHECK:         BinaryOperator {{.*}} 'float' '+' ConstRoundingMode=dynamic

#pragma STDC FENV_ROUND FE_TONEAREST

float func_13(float x, float y) {
  return x + y;
}

// CHECK-LABEL: FunctionDecl {{.*}} func_13 'float (float, float)'
// CHECK:         BinaryOperator {{.*}} 'float' '+' ConstRoundingMode=tonearest


template <typename T>
T func_14(T x, T y) {
#pragma STDC FENV_ROUND FE_TOWARDZERO
  return x + y;
}

float func_15(float x, float y) {
#pragma STDC FENV_ROUND FE_DOWNWARD
  return func_14(x, y);
}

// CHECK-LABEL: FunctionTemplateDecl {{.*}} func_14
// CHECK:         FunctionDecl {{.*}} func_14 'T (T, T)'
// CHECK:           CompoundStmt
// CHECK-NEXT:        ReturnStmt
// CHECK-NEXT:          BinaryOperator {{.*}} '+' ConstRoundingMode=towardzero
// CHECK:         FunctionDecl {{.*}} func_14 'float (float, float)'
// CHECK:           CompoundStmt
// CHECK-NEXT:        ReturnStmt
// CHECK-NEXT:          BinaryOperator {{.*}} 'float' '+' ConstRoundingMode=towardzero

float func_16(float x, float y) {
#pragma STDC FENV_ROUND FE_TOWARDZERO
  if (x < 0) {
#pragma STDC FENV_ROUND FE_UPWARD
    return x - y;
  }
  return x + y;
}

// CHECK-LABEL: FunctionDecl {{.*}} func_16 'float (float, float)'
// CHECK:         CompoundStmt {{.*}} ConstRoundingMode=towardzero
// CHECK:           IfStmt
// CHECK:             CompoundStmt {{.*}} ConstRoundingMode=upward
// CHECK:               ReturnStmt
// CHECK:                 BinaryOperator {{.*}} ConstRoundingMode=upward
// CHECK:           ReturnStmt
// CHECK:             BinaryOperator {{.*}} ConstRoundingMode=towardzero

float func_17(float x, float y) {
#pragma STDC FENV_ROUND FE_TOWARDZERO
  if (x < 0) {
#pragma STDC FENV_ROUND FE_TOWARDZERO
    return x - y;
  }
  return x + y;
}

// CHECK-LABEL: FunctionDecl {{.*}} func_17 'float (float, float)'
// CHECK:         CompoundStmt {{.*}} ConstRoundingMode=towardzero
// CHECK:           IfStmt
// CHECK:             CompoundStmt {{.*}}
// CHECK:               ReturnStmt
// CHECK:                 BinaryOperator {{.*}} ConstRoundingMode=towardzero
// CHECK:           ReturnStmt
// CHECK:             BinaryOperator {{.*}} ConstRoundingMode=towardzero

#pragma STDC FENV_ROUND FE_DOWNWARD
float func_18(float x, float y) {
  return x + y;
}

// CHECK-LABEL: FunctionDecl {{.*}} func_18 'float (float, float)'
// CHECK:         CompoundStmt {{.*}} ConstRoundingMode=downward
// CHECK:           ReturnStmt
// CHECK:             BinaryOperator {{.*}} ConstRoundingMode=downward

#pragma float_control(precise, off)
__attribute__((optnone))
float func_19(float x, float y) {
  return x + y;
}

// CHECK-LABEL: FunctionDecl {{.*}} func_19 'float (float, float)'
// CHECK:         CompoundStmt {{.*}} MathErrno=1
// CHECK:           ReturnStmt
// CHECK:             BinaryOperator {{.*}} 'float' '+' ConstRoundingMode=downward MathErrno=1
