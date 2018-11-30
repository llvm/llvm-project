// RUN: %clang_cc1 -std=c11 -ast-dump -ast-dump-filter Test %s | FileCheck -strict-whitespace %s

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

void TestGenericSelectionExpressions(int i) {
  _Generic(i, int : 12);
  // CHECK: GenericSelectionExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:23> 'int'
  // CHECK-NEXT: ImplicitCastExpr 0x{{[^ ]*}}
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:12> 'int' lvalue ParmVar 0x{{[^ ]*}} 'i' 'int'
  // CHECK-NEXT: BuiltinType 0x{{[^ ]*}} 'int'
  // FIXME: note that the following test line has a spurious whitespace.
  // CHECK-NEXT: case  'int' selected
  // CHECK-NEXT: BuiltinType 0x{{[^ ]*}} 'int'
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:21> 'int' 12
  _Generic(i, int : 12, default : 0);
  // CHECK: GenericSelectionExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:36> 'int'
  // CHECK-NEXT: ImplicitCastExpr 0x{{[^ ]*}}
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:12> 'int' lvalue ParmVar 0x{{[^ ]*}} 'i' 'int'
  // CHECK-NEXT: BuiltinType 0x{{[^ ]*}} 'int'
  // FIXME: note that the following test line has a spurious whitespace.
  // CHECK-NEXT: case  'int' selected
  // CHECK-NEXT: BuiltinType 0x{{[^ ]*}} 'int'
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:21> 'int' 12
  // CHECK-NEXT: default
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:35> 'int' 0
  _Generic(i, default : 0, int : 12);
  // CHECK: GenericSelectionExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:36> 'int'
  // CHECK-NEXT: ImplicitCastExpr 0x{{[^ ]*}}
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:12> 'int' lvalue ParmVar 0x{{[^ ]*}} 'i' 'int'
  // CHECK-NEXT: BuiltinType 0x{{[^ ]*}} 'int'
  // CHECK-NEXT: default
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:25> 'int' 0
  // FIXME: note that the following test line has a spurious whitespace.
  // CHECK-NEXT: case  'int' selected
  // CHECK-NEXT: BuiltinType 0x{{[^ ]*}} 'int'
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:34> 'int' 12
  _Generic(i, int : 12, float : 10, default : 100);
  // CHECK: GenericSelectionExpr 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:50> 'int'
  // CHECK-NEXT: ImplicitCastExpr 0x{{[^ ]*}}
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:12> 'int' lvalue ParmVar 0x{{[^ ]*}} 'i' 'int'
  // CHECK-NEXT: BuiltinType 0x{{[^ ]*}} 'int'
  // FIXME: note that the following test line has a spurious whitespace.
  // CHECK-NEXT: case  'int' selected
  // CHECK-NEXT: BuiltinType 0x{{[^ ]*}} 'int'
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:21> 'int' 12
  // FIXME: note that the following test line has a spurious whitespace.
  // CHECK-NEXT: case  'float'
  // CHECK-NEXT: BuiltinType 0x{{[^ ]*}} 'float'
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:33> 'int' 10
  // CHECK-NEXT: default
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:47> 'int' 100

  int j = _Generic(i, int : 12);
  // CHECK: DeclStmt 0x{{[^ ]*}} <line:[[@LINE-1]]:3, col:32>
  // CHECK-NEXT: VarDecl 0x{{[^ ]*}} <col:3, col:31> col:7 j 'int' cinit
  // CHECK-NEXT: GenericSelectionExpr 0x{{[^ ]*}} <col:11, col:31> 'int'
  // CHECK-NEXT: ImplicitCastExpr 0x{{[^ ]*}}
  // CHECK-NEXT: DeclRefExpr 0x{{[^ ]*}} <col:20> 'int' lvalue ParmVar 0x{{[^ ]*}} 'i' 'int'
  // CHECK-NEXT: BuiltinType 0x{{[^ ]*}} 'int'
  // FIXME: note that the following test line has a spurious whitespace.
  // CHECK-NEXT: case  'int' selected
  // CHECK-NEXT: BuiltinType 0x{{[^ ]*}} 'int'
  // CHECK-NEXT: IntegerLiteral 0x{{[^ ]*}} <col:29> 'int' 12
}
