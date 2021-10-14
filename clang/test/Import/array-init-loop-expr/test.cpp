// RUN: clang-import-test -dump-ast -import %S/Inputs/S.cpp -expression %s | FileCheck %s
// CHECK: CXXCtorInitializer
// CHECK-NEXT: ArrayInitLoopExpr
// CHECK-SAME: 'int [10]'

// CHECK: ArrayInitIndexExpr

void expr() {
  f();
}
