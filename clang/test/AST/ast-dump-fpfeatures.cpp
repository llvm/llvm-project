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