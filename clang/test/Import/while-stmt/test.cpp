// RUN: clang-import-test -dump-ast -import %S/Inputs/F.cpp -expression %s | FileCheck %s

// CHECK: WhileStmt
// CHECK-NEXT: <<NULL>>
// CHECK-NEXT: CXXBoolLiteralExpr
// CHECK-NEXT: NullStmt

// CHECK: WhileStmt
// CHECK-NEXT: <<NULL>>
// CHECK-NEXT: CXXBoolLiteralExpr
// CHECK-NEXT: CompoundStmt

// CHECK: WhileStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl
// CHECK-NEXT: CXXBoolLiteralExpr
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr
// CHECK-NEXT: NullStmt

void expr() {
  f();
}
