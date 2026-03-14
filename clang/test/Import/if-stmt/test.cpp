// RUN: clang-import-test -dump-ast -import %S/Inputs/F.cpp -expression %s | FileCheck %s

// CHECK: IfStmt
// CHECK-NEXT: CXXBoolLiteralExpr
// CHECK-NEXT: ReturnStmt

// CHECK: IfStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl
// CHECK-NEXT: IntegerLiteral
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr
// CHECK-NEXT: ReturnStmt

// CHECK: IfStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl
// CHECK-NEXT: CXXBoolLiteralExpr
// CHECK-NEXT: ReturnStmt

// CHECK: IfStmt
// CHECK-NEXT: CXXBoolLiteralExpr
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: ReturnStmt

// CHECK: IfStmt
// CHECK-NEXT: CXXBoolLiteralExpr
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt

void expr() {
  f();
}
