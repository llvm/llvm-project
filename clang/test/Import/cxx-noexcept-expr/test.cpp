// RUN: clang-import-test -dump-ast -import %S/Inputs/F.cpp -expression %s | FileCheck %s

// CHECK: CXXNoexceptExpr
// CHECK-NEXT: IntegerLiteral

void expr() {
  f();
}
