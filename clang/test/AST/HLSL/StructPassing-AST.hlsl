// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -ast-dump -o - %s | FileCheck %s

// CHECK: CXXRecordDecl {{.*}} struct P definition
// CHECK-NEXT: DefinitionData aggregate standard_layout trivially_copyable pod literal can_const_default_init
// CHECK-NOT: DefaultConstructor {{.*}} exists
// CHECK-NOT: MoveConstructor {{.*}} exists
// CHECK-NOT: MoveAssignment {{.*}} exists
// CHECK: PackedAttr
// CHECK-NEXT: CXXRecordDecl {{.*}} struct P
// CHECK-NEXT: FieldDecl {{.*}} a 'float'
// CHECK-NOT: CXXConstructorDecl
// CHECK-NOT: CXXMethodDecl {{.*}} operator=
struct P {
  float a;
};

// CHECK: CXXRecordDecl {{.*}} struct S definition
// CHECK-NEXT: DefinitionData aggregate trivially_copyable literal can_const_default_init
// CHECK-NOT: DefaultConstructor {{.*}} exists
// CHECK-NOT: MoveConstructor {{.*}} exists
// CHECK-NOT: MoveAssignment {{.*}} exists
// CHECK: public 'P'
// CHECK-NEXT: PackedAttr
// CHECK-NEXT: CXXRecordDecl {{.*}} implicit struct S
// CHECK-NEXT: FieldDecl {{.*}} b 'double'
// CHECK-NEXT: FieldDecl {{.*}} c 'int[2]'
// CHECK-NOT: CXXConstructorDecl
// CHECK-NOT: CXXMethodDecl {{.*}} operator=
struct S : P {
  double b;
  int c[2];
};

// CHECK: FunctionDecl {{.*}} case1 'void (S)'
// CHECK-NEXT: ParmVarDecl {{.*}} s 'S'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} sLocal 'S'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'S' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'S' lvalue ParmVar {{.*}} 's' 'S'
void case1(S s) {
  // struct initialization
  S sLocal = s;
}

// CHECK: FunctionDecl {{.*}} case2 'void (S)'
// CHECK-NEXT: ParmVarDecl {{.*}} s 'S'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} sLocal 'S'
// CHECK-NEXT: BinaryOperator {{.*}} 'S' lvalue '='
// CHECK-NEXT: DeclRefExpr {{.*}} 'S' lvalue Var {{.*}} 'sLocal' 'S'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'S' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'S' lvalue ParmVar {{.*}} 's' 'S' 
void case2(S s) {
  S sLocal;
  // struct assignment
  sLocal = s;
}

void useS(S s) {}

// CHECK: FunctionDecl {{.*}} case3 'void (S)'
// CHECK-NEXT: ParmVarDecl {{.*}} used s 'S'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: CallExpr {{.*}} 'void'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(S)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'void (S)' lvalue Function {{.*}} 'useS' 'void (S)'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'S' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'S' lvalue ParmVar {{.*}} 's' 'S'
void case3(S s) {
  // struct argument passing
  useS(s);
}

// CHECK: FunctionDecl {{.*}} case4 'void (S)'
// CHECK-NEXT: ParmVarDecl {{.*}} used s 'S'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} pLocal 'P'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'P' <DerivedToBase (P)>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'S' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'S' lvalue ParmVar {{.*}} 's' 'S'
void case4(S s) {
  // derived to base conversion in initialization
  P pLocal = s;
}

// CHECK: FunctionDecl {{.*}} case5 'void (S)'
// CHECK-NEXT: ParmVarDecl {{.*}} used s 'S'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} pLocal 'P'
// CHECK-NEXT: BinaryOperator {{.*}} 'P' lvalue '='
// CHECK-NEXT: DeclRefExpr {{.*}} 'P' lvalue Var {{.*}} 'pLocal' 'P'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'P' <LValueToRValue>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'P' lvalue <DerivedToBase (P)>
// CHECK-NEXT: DeclRefExpr {{.*}} 'S' lvalue ParmVar {{.*}} 's' 'S'
void case5(S s) {
  P pLocal;
  // derived to base conversion in assignment
  pLocal = s;
}

void useP(P p) {}

// CHECK: FunctionDecl {{.*}} case6 'void (S)'
// CHECK-NEXT: ParmVarDecl {{.*}} used s 'S'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: CallExpr {{.*}} 'void'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(P)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'void (P)' lvalue Function {{.*}} 'useP' 'void (P)'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'P' <DerivedToBase (P)>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'S' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'S' lvalue ParmVar {{.*}} 's' 'S'
void case6(S s) {
  // derived to base conversion in argument passing
  useP(s);
}

struct G {
  int X;
  void f() {}
  void setX(int val) { X = val; }
  int getX() const { return X; }
};

groupshared G g;
  
// CHECK: FunctionDecl {{.*}} case7 'void ()' external-linkage
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} '<bound member function type>' .f
// CHECK-NEXT: HLSLOutArgExpr {{.*}} 'G' lvalue inout
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'G' lvalue
// CHECK-NEXT: DeclRefExpr {{.*}} 'groupshared G' lvalue Var {{.*}} 'g' 'groupshared G'
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'G' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'G' <LValueToRValue>
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'G' lvalue
// CHECK-NEXT: DeclRefExpr {{.*}} 'groupshared G' lvalue Var {{.*}} 'g' 'groupshared G'
// CHECK-NEXT: BinaryOperator {{.*}} 'G' lvalue '='
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'G' lvalue
// CHECK-NEXT: DeclRefExpr {{.*}} 'groupshared G' lvalue Var {{.*}} 'g' 'groupshared G'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'G' <LValueToRValue>
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'G' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'G' <LValueToRValue>
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'G' lvalue
// CHECK-NEXT: DeclRefExpr {{.*}} 'groupshared G' lvalue Var {{.*}} 'g' 'groupshared G'
void case7() {
  g.f();
}

// CHECK: FunctionDecl {{.*}} case8 'void (int)' external-linkage
// CHECK-NEXT: ParmVarDecl {{.*}} used V 'int'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} '<bound member function type>' .setX
// CHECK-NEXT: HLSLOutArgExpr {{.*}} 'G' lvalue inout
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'G' lvalue
// CHECK-NEXT: DeclRefExpr {{.*}} 'groupshared G' lvalue Var {{.*}} 'g' 'groupshared G'
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'G' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'G' <LValueToRValue>
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'G' lvalue
// CHECK-NEXT: DeclRefExpr {{.*}} 'groupshared G' lvalue Var {{.*}} 'g' 'groupshared G'
// CHECK-NEXT: BinaryOperator {{.*}} 'G' lvalue '='
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'G' lvalue
// CHECK-NEXT: DeclRefExpr {{.*}} 'groupshared G' lvalue Var {{.*}} 'g' 'groupshared G'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'G' <LValueToRValue>
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'G' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'G' <LValueToRValue>
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'G' lvalue
// CHECK-NEXT: DeclRefExpr {{.*}} 'groupshared G' lvalue Var {{.*}} 'g' 'groupshared G'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'V' 'int'
void case8(int V) {
  g.setX(V);
}

// CHECK: FunctionDecl {{.*}} case9 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} Y 'int' cinit
// CHECK-NEXT: ExprWithCleanups {{.*}} 'int'
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'int'
// CHECK-NEXT: MemberExpr {{.*}} '<bound member function type>' .getX
// CHECK-NEXT: MaterializeTemporaryExpr {{.*}} 'const G' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const G' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'groupshared G' lvalue Var {{.*}} 'g' 'groupshared G'
void case9() {
  int Y = g.getX();
}

// CHECK-NOT: CXXConstructExpr
// CHECK-NOT: CXXOperatorCallExpr
