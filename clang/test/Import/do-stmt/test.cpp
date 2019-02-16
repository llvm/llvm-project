// RUN: clang-import-test -dump-ast -import %S/Inputs/F.cpp -expression %s | FileCheck %s

// CHECK: DoStmt
// CHECK-NEXT: NullStmt
// CHECK-NEXT: CXXBoolLiteralExpr
// CHECK-SAME: true

// CHECK: DoStmt
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: CXXBoolLiteralExpr
// CHECK-SAME: false

void expr() {
  f();
}
