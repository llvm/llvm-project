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
