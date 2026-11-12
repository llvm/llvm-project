// RUN: clang-import-test -dump-ast -import %S/Inputs/S.cpp -expression %s | FileCheck %s
// CHECK: ExprWithCleanups
// CHECK-SAME: 'RAII'
// CHECK-NEXT: CXXBindTemporaryExpr

void expr() {
  f();
}
