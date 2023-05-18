// RUN: clang-import-test -dump-ast -import %S/Inputs/F.cpp -expression %s | FileCheck %s

// CHECK: ForStmt
// CHECK-NEXT: <<NULL>>
// CHECK-NEXT: <<NULL>>
// CHECK-NEXT: <<NULL>>
// CHECK-NEXT: <<NULL>>
// CHECK-NEXT: NullStmt

// CHECK: ForStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl
// CHECK-NEXT: IntegerLiteral
// CHECK-NEXT: <<NULL>>
// CHECK-NEXT: <<NULL>>
// CHECK-NEXT: <<NULL>>
// CHECK-NEXT: ContinueStmt

// CHECK: ForStmt
// CHECK-NEXT: <<NULL>>
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl
// CHECK-NEXT: CXXBoolLiteralExpr
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr
// CHECK-SAME: 'j'
// CHECK-SAME: 'bool'
// CHECK-NEXT: <<NULL>>
// CHECK-NEXT: ContinueStmt

// CHECK: ForStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl
// CHECK-NEXT: IntegerLiteral
// CHECK-NEXT: <<NULL>>

// CHECK-NEXT: BinaryOperator
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr
// CHECK-NEXT: IntegerLiteral

// CHECK-NEXT: UnaryOperator
// CHECK-SAME: '++'
// CHECK-NEXT: DeclRefExpr

// CHECK-NEXT: CompoundStmt

void expr() {
  f();
}
