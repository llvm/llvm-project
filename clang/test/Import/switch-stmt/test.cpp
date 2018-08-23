// RUN: clang-import-test -dump-ast -import %S/Inputs/F.cpp -expression %s | FileCheck %s

// CHECK: SwitchStmt
// CHECK-NEXT: <<NULL>>
// CHECK-NEXT: <<NULL>>
// CHECK-NEXT: IntegerLiteral
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: CaseStmt
// CHECK-NEXT: IntegerLiteral
// CHECK-NEXT: <<NULL>>
// CHECK-NEXT: CaseStmt
// CHECK-NEXT: IntegerLiteral
// CHECK-NEXT: <<NULL>>
// CHECK-NEXT: BreakStmt

// CHECK: SwitchStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl
// CHECK-SAME: varname
// CHECK-NEXT: <<NULL>>
// CHECK-NEXT: IntegerLiteral
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: CaseStmt
// CHECK-NEXT: IntegerLiteral
// CHECK-NEXT: <<NULL>>
// CHECK-NEXT: BreakStmt
// CHECK-NEXT: CaseStmt
// CHECK-NEXT: IntegerLiteral
// CHECK-NEXT: <<NULL>>
// CHECK-NEXT: BreakStmt

// CHECK: SwitchStmt
// CHECK-NEXT: <<NULL>>
// CHECK-NEXT: <<NULL>>
// CHECK-NEXT: IntegerLiteral
// CHECK-NEXT: DefaultStmt
// CHECK-NEXT: BreakStmt

// CHECK: SwitchStmt
// CHECK-NEXT: <<NULL>>
// CHECK-NEXT: <<NULL>>
// CHECK-NEXT: IntegerLiteral
// CHECK-NEXT: NullStmt

void expr() {
  f();
}
