// RUN: %clang_cc1 -ast-dump -ast-dump-filter Test %s | FileCheck -strict-whitespace %s

int TestLocation = 0;
// CHECK:      Dumping TestLocation
// CHECK-NEXT:   IntegerLiteral 0x{{[^ ]*}} <{{.*}}:3:20> 'int' 0

int TestIndent = 1 + (1);
// CHECK:      Dumping TestIndent
// CHECK-NEXT: {{\(BinaryOperator[^()]*$}}
// CHECK-NEXT: {{^  \(IntegerLiteral.*0[^()]*\)$}}
// CHECK-NEXT: {{^  \(ParenExpr.*0[^()]*$}}
// CHECK-NEXT: {{^    \(IntegerLiteral.*0[^()]*\)\)\)$}}

void TestDeclStmt() {
  int x = 0;
  int y, z;
}
// CHECK:      Dumping TestDeclStmt
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT:   DeclStmt
// CHECK-NEXT:     int x =
// CHECK-NEXT:       IntegerLiteral
// CHECK-NEXT:   DeclStmt
// CHECK-NEXT:     int y
// CHECK-NEXT:     int z

int TestOpaqueValueExpr = 0 ?: 1;
// CHECK:      Dumping TestOpaqueValueExpr
// CHECK-NEXT: BinaryConditionalOperator
// CHECK-NEXT:   IntegerLiteral
// CHECK-NEXT:   OpaqueValueExpr
// CHECK-NEXT:     IntegerLiteral
// CHECK-NEXT:   OpaqueValueExpr
// CHECK-NEXT:     IntegerLiteral
// CHECK-NEXT:   IntegerLiteral
