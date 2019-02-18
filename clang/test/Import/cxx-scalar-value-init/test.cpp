// RUN: clang-import-test -dump-ast -import %S/Inputs/S.cpp -expression %s | FileCheck %s
// CHECK: CXXScalarValueInitExpr
// CHECK-SAME: 'int'

// CHECK: CXXScalarValueInitExpr
// CHECK-SAME: 'float'

void expr() {
  int i = si();
  float f = sf();
}
