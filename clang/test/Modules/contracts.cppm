// Tests that contract annotations survive module serialization/deserialization.
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++2c -fcontracts %t/A.cppm -emit-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++2c -fcontracts %t/Use.cpp -fprebuilt-module-path=%t \
// RUN:   -fsyntax-only -ast-dump-all 2>&1 | FileCheck %t/Use.cpp

// Test again with reduced BMI.
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++2c -fcontracts %t/A.cppm -emit-reduced-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++2c -fcontracts %t/Use.cpp -fprebuilt-module-path=%t \
// RUN:   -fsyntax-only -ast-dump-all 2>&1 | FileCheck %t/Use.cpp

//--- A.cppm
export module A;

export int divide(int a, int b) pre(b != 0);

export int clamp(int x, int lo, int hi)
    pre(lo <= hi)
    post(x >= lo);

export int square(int x) post(r: r >= 0);

//--- Use.cpp
import A;

int test() {
  return divide(10, 2) + clamp(5, 0, 10) + square(3);
}

// CHECK: FunctionDecl {{.*}} divide 'int (int, int)' {{.*}}contracts
// CHECK:   pre:
// CHECK:     BinaryOperator {{.*}} '!='

// CHECK: FunctionDecl {{.*}} clamp 'int (int, int, int)' {{.*}}contracts
// CHECK:   pre:
// CHECK:     BinaryOperator {{.*}} '<='
// CHECK:   post:
// CHECK:     BinaryOperator {{.*}} '>='

// CHECK: FunctionDecl {{.*}} square 'int (int)' {{.*}}contracts
// CHECK:   post:
// CHECK:     VarDecl {{.*}} implicit {{.*}} r 'const int'
// CHECK:     BinaryOperator {{.*}} '>='
// CHECK:       DeclRefExpr {{.*}} 'const int' lvalue Var {{.*}} 'r' 'const int'
