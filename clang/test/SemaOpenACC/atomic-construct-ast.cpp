// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -fopenacc -emit-pch -o %t %s
// RUN: %clang_cc1 %s -fopenacc -include-pch %t -ast-dump-all | FileCheck %s

#ifndef PCH_HELPER
#define PCH_HELPER

void foo(int v, int x) {
  // CHECK: FunctionDecl{{.*}} foo 'void (int, int)'
  // CHECK-NEXT: ParmVarDecl
  // CHECK-NEXT: ParmVarDecl
  // CHECK-NEXT: CompoundStmt

// CHECK-NEXT: OpenACCAtomicConstruct{{.*}} atomic read
// CHECK-NEXT: BinaryOperator{{.*}} 'int' lvalue '='
// CHECK-NEXT: DeclRefExpr{{.*}}'v' 'int'
// CHECK-NEXT: ImplicitCastExpr{{.*}}'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr{{.*}}'x' 'int'
#pragma acc atomic read
  v = x;

// CHECK-NEXT: OpenACCAtomicConstruct{{.*}} atomic write
// CHECK-NEXT: BinaryOperator{{.*}} 'int' lvalue '='
// CHECK-NEXT: DeclRefExpr{{.*}}'v' 'int'
// CHECK-NEXT: BinaryOperator{{.*}}'int' '+'
// CHECK-NEXT: ImplicitCastExpr{{.*}}'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr{{.*}}'x' 'int'
// CHECK-NEXT: IntegerLiteral{{.*}} 'int' 1
#pragma acc atomic write
  v = x + 1;

// CHECK-NEXT: OpenACCAtomicConstruct{{.*}} atomic update 
// CHECK-NEXT: UnaryOperator{{.*}} 'int' postfix '++'
// CHECK-NEXT: DeclRefExpr{{.*}}'x' 'int'
#pragma acc atomic update
  x++;
// CHECK-NEXT: OpenACCAtomicConstruct{{.*}} atomic <none>
// CHECK-NEXT: UnaryOperator{{.*}} 'int' postfix '--'
// CHECK-NEXT: DeclRefExpr{{.*}}'x' 'int'
#pragma acc atomic
  x--;
// CHECK-NEXT: OpenACCAtomicConstruct{{.*}} atomic capture
// CHECK-NEXT: BinaryOperator{{.*}} 'int' lvalue '='
// CHECK-NEXT: DeclRefExpr{{.*}}'v' 'int'
// CHECK-NEXT: UnaryOperator{{.*}} 'int' postfix '++'
// CHECK-NEXT: DeclRefExpr{{.*}}'x' 'int'
#pragma acc atomic capture
  v = x++;

// CHECK-NEXT: OpenACCAtomicConstruct{{.*}} atomic capture
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: UnaryOperator{{.*}} 'int' postfix '--'
// CHECK-NEXT: DeclRefExpr{{.*}}'x' 'int'
// CHECK-NEXT: BinaryOperator{{.*}} 'int' lvalue '='
// CHECK-NEXT: DeclRefExpr{{.*}}'v' 'int'
// CHECK-NEXT: ImplicitCastExpr{{.*}}'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr{{.*}}'x' 'int'
#pragma acc atomic capture
  { x--; v = x; }

}

template<typename T, int I>
void templ_foo(T v, T x) {
  // CHECK-NEXT: FunctionTemplateDecl{{.*}}templ_foo
  // CHECK-NEXT: TemplateTypeParmDecl{{.*}} T
  // CHECK-NEXT: NonTypeTemplateParmDecl{{.*}} I
  // CHECK-NEXT: FunctionDecl{{.*}} templ_foo 'void (T, T)'
  // CHECK-NEXT: ParmVarDecl{{.*}} v 'T'
  // CHECK-NEXT: ParmVarDecl{{.*}} x 'T'
  // CHECK-NEXT: CompoundStmt

// CHECK-NEXT: OpenACCAtomicConstruct{{.*}} atomic read
// CHECK-NEXT: BinaryOperator{{.*}} '<dependent type>' '='
// CHECK-NEXT: DeclRefExpr{{.*}}'v' 'T'
// CHECK-NEXT: DeclRefExpr{{.*}}'x' 'T'
#pragma acc atomic read
  v = x;

// CHECK-NEXT: OpenACCAtomicConstruct{{.*}} atomic write
// CHECK-NEXT: BinaryOperator{{.*}} '<dependent type>' '='
// CHECK-NEXT: DeclRefExpr{{.*}}'v' 'T'
// CHECK-NEXT: BinaryOperator{{.*}}'<dependent type>' '+'
// CHECK-NEXT: DeclRefExpr{{.*}}'x' 'T'
// CHECK-NEXT: DeclRefExpr{{.*}} 'I' 'int'
#pragma acc atomic write
  v = x + I;

// CHECK-NEXT: OpenACCAtomicConstruct{{.*}} atomic update 
// CHECK-NEXT: UnaryOperator{{.*}} '<dependent type>' postfix '++'
// CHECK-NEXT: DeclRefExpr{{.*}}'x' 'T'
#pragma acc atomic update
  x++;
// CHECK-NEXT: OpenACCAtomicConstruct{{.*}} atomic <none>
// CHECK-NEXT: UnaryOperator{{.*}} '<dependent type>' postfix '--'
// CHECK-NEXT: DeclRefExpr{{.*}}'x' 'T'
#pragma acc atomic
  x--;
// CHECK-NEXT: OpenACCAtomicConstruct{{.*}} atomic capture
// CHECK-NEXT: BinaryOperator{{.*}} '<dependent type>' '='
// CHECK-NEXT: DeclRefExpr{{.*}}'v' 'T'
// CHECK-NEXT: UnaryOperator{{.*}} '<dependent type>' postfix '++'
// CHECK-NEXT: DeclRefExpr{{.*}}'x' 'T'
#pragma acc atomic capture
  v = x++;

// CHECK-NEXT: OpenACCAtomicConstruct{{.*}} atomic capture
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: UnaryOperator{{.*}} '<dependent type>' postfix '--'
// CHECK-NEXT: DeclRefExpr{{.*}}'x' 'T'
// CHECK-NEXT: BinaryOperator{{.*}} '<dependent type>' '='
// CHECK-NEXT: DeclRefExpr{{.*}}'v' 'T'
// CHECK-NEXT: DeclRefExpr{{.*}}'x' 'T'
#pragma acc atomic capture
  { x--; v = x; }

  // CHECK-NEXT: FunctionDecl{{.*}} templ_foo 'void (int, int)' implicit_instantiation
  // CHECK-NEXT: TemplateArgument type 'int'
  // CHECK-NEXT: BuiltinType{{.*}} 'int'
  // CHECK-NEXT: TemplateArgument integral '5'
  // CHECK-NEXT: ParmVarDecl{{.*}} v 'int'
  // CHECK-NEXT: ParmVarDecl{{.*}} x 'int'
  // CHECK-NEXT: CompoundStmt

// CHECK-NEXT: OpenACCAtomicConstruct{{.*}} atomic read
// CHECK-NEXT: BinaryOperator{{.*}} 'int' lvalue '='
// CHECK-NEXT: DeclRefExpr{{.*}}'v' 'int'
// CHECK-NEXT: ImplicitCastExpr{{.*}}'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr{{.*}}'x' 'int'

// CHECK-NEXT: OpenACCAtomicConstruct{{.*}} atomic write
// CHECK-NEXT: BinaryOperator{{.*}} 'int' lvalue '='
// CHECK-NEXT: DeclRefExpr{{.*}}'v' 'int'
// CHECK-NEXT: BinaryOperator{{.*}}'int' '+'
// CHECK-NEXT: ImplicitCastExpr{{.*}}'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr{{.*}}'x' 'int'
// CHECK-NEXT: SubstNonTypeTemplateParmExpr{{.*}} 'int'
// CHECK-NEXT: NonTypeTemplateParmDecl{{.*}} 'int'{{.*}}I
// CHECK-NEXT: IntegerLiteral{{.*}} 'int' 5

// CHECK-NEXT: OpenACCAtomicConstruct{{.*}} atomic update 
// CHECK-NEXT: UnaryOperator{{.*}} 'int' postfix '++'
// CHECK-NEXT: DeclRefExpr{{.*}}'x' 'int'

// CHECK-NEXT: OpenACCAtomicConstruct{{.*}} atomic <none>
// CHECK-NEXT: UnaryOperator{{.*}} 'int' postfix '--'
// CHECK-NEXT: DeclRefExpr{{.*}}'x' 'int'

// CHECK-NEXT: OpenACCAtomicConstruct{{.*}} atomic capture
// CHECK-NEXT: BinaryOperator{{.*}} 'int' lvalue '='
// CHECK-NEXT: DeclRefExpr{{.*}}'v' 'int'
// CHECK-NEXT: UnaryOperator{{.*}} 'int' postfix '++'
// CHECK-NEXT: DeclRefExpr{{.*}}'x' 'int'

// CHECK-NEXT: OpenACCAtomicConstruct{{.*}} atomic capture
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: UnaryOperator{{.*}} 'int' postfix '--'
// CHECK-NEXT: DeclRefExpr{{.*}}'x' 'int'
// CHECK-NEXT: BinaryOperator{{.*}} 'int' lvalue '='
// CHECK-NEXT: DeclRefExpr{{.*}}'v' 'int'
// CHECK-NEXT: ImplicitCastExpr{{.*}}'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr{{.*}}'x' 'int'
}

void use() {
  templ_foo<int, 5>(1, 2);
}
#endif
