// RUN: clang-import-test -dump-ast -import %S/Inputs/F.cpp -expression %s | FileCheck %s

// CHECK: CXXForRangeStmt

// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl
// CHECK-NEXT: DeclRefExpr
// CHECK-SAME: 'c'
// CHECK-SAME: Container

// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl
// CHECK-NEXT: CXXMemberCallExpr
// CHECK-SAME: 'int *'
// CHECK-NEXT: MemberExpr
// CHECK-SAME: .begin
// CHECK-NEXT: DeclRefExpr
// CHECK-SAME: '__range1'
// CHECK-SAME: Container

// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl
// CHECK-NEXT: CXXMemberCallExpr
// CHECK-SAME: 'int *'
// CHECK-NEXT: MemberExpr
// CHECK-SAME: .end
// CHECK-NEXT: DeclRefExpr
// CHECK-SAME: '__range1'
// CHECK-SAME: Container

// CHECK-NEXT: BinaryOperator
// CHECK-SAME: '!='
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr
// CHECK-SAME: '__begin1'
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr
// CHECK-SAME: '__end1'

// CHECK-NEXT: UnaryOperator
// CHECK-SAME: '++'
// CHECK-NEXT: DeclRefExpr
// CHECK-SAME: '__begin1'

// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl
// CHECK-SAME: varname

// CHECK: ReturnStmt

void expr() {
  f();
}
