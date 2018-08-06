// RUN: %clang_cc1 -ast-dump -ast-dump-filter Test %s | FileCheck -strict-whitespace %s

int TestLocation = 0;
// CHECK:      VarDecl{{.*}}TestLocation
// CHECK-NEXT:   IntegerLiteral 0x{{[^ ]*}} <col:20> 'int' 0

int TestIndent = 1 + (1);
// CHECK:      VarDecl{{.*}}TestIndent
// CHECK-NEXT: {{^}}`-BinaryOperator{{[^()]*$}}
// CHECK-NEXT: {{^}}  |-IntegerLiteral{{.*0[^()]*$}}
// CHECK-NEXT: {{^}}  `-ParenExpr{{.*0[^()]*$}}
// CHECK-NEXT: {{^}}    `-IntegerLiteral{{.*0[^()]*$}}

void TestDeclStmt() {
  int x = 0;
  int y, z;
}
// CHECK:      FunctionDecl{{.*}}TestDeclStmt
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT:   DeclStmt
// CHECK-NEXT:     VarDecl{{.*}}x
// CHECK-NEXT:       IntegerLiteral
// CHECK-NEXT:   DeclStmt
// CHECK-NEXT:     VarDecl{{.*}}y
// CHECK-NEXT:     VarDecl{{.*}}z

int TestOpaqueValueExpr = 0 ?: 1;
// CHECK:      VarDecl{{.*}}TestOpaqueValueExpr
// CHECK-NEXT: BinaryConditionalOperator
// CHECK-NEXT:   IntegerLiteral
// CHECK-NEXT:   OpaqueValueExpr
// CHECK-NEXT:     IntegerLiteral
// CHECK-NEXT:   OpaqueValueExpr
// CHECK-NEXT:     IntegerLiteral
// CHECK-NEXT:   IntegerLiteral

void TestUnaryOperatorExpr(void) {
  char T1 = 1;
  int T2 = 1;

  T1++;
  T2++;
  // CHECK:      UnaryOperator{{.*}}postfix '++' cannot overflow
  // CHECK-NEXT:   DeclRefExpr{{.*}}'T1' 'char'
  // CHECK-NOT:  UnaryOperator{{.*}}postfix '++' cannot overflow
  // CHECK:        DeclRefExpr{{.*}}'T2' 'int'

  -T1;
  -T2;
  // CHECK:      UnaryOperator{{.*}}prefix '-' cannot overflow
  // CHECK-NEXT:   ImplicitCastExpr
  // CHECK-NEXT:     ImplicitCastExpr
  // CHECK-NEXT:       DeclRefExpr{{.*}}'T1' 'char'
  // CHECK-NOT:  UnaryOperator{{.*}}prefix '-' cannot overflow
  // CHECK:        ImplicitCastExpr
  // CHECK:          DeclRefExpr{{.*}}'T2' 'int'

  ~T1;
  ~T2;
  // CHECK:      UnaryOperator{{.*}}prefix '~' cannot overflow
  // CHECK-NEXT:   ImplicitCastExpr
  // CHECK-NEXT:     ImplicitCastExpr
  // CHECK-NEXT:       DeclRefExpr{{.*}}'T1' 'char'
  // CHECK:  	 UnaryOperator{{.*}}prefix '~' cannot overflow
  // CHECK-NEXT:     ImplicitCastExpr
  // CHECK-NEXT:       DeclRefExpr{{.*}}'T2' 'int'
}
