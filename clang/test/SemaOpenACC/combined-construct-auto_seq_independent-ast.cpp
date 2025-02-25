// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -fopenacc -emit-pch -o %t %s
// RUN: %clang_cc1 %s -fopenacc -include-pch %t -ast-dump-all | FileCheck %s
#ifndef PCH_HELPER
#define PCH_HELPER

void NormalUses() {
  // CHECK: FunctionDecl{{.*}}NormalUses
  // CHECK-NEXT: CompoundStmt

#pragma acc parallel loop auto
  for(int i = 0; i < 5; ++i){}
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} parallel loop
  // CHECK-NEXT: auto clause
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} i 'int'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 0
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: BinaryOperator{{.*}}'<'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'i' 'int'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 5
  // CHECK-NEXT: UnaryOperator{{.*}}++
  // CHECK-NEXT: DeclRefExpr{{.*}}'i' 'int'
  // CHECK-NEXT: CompoundStmt

#pragma acc serial loop seq
  for(;;){}
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} serial loop
  // CHECK-NEXT: seq clause
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: CompoundStmt

#pragma acc kernels loop independent
  for(int i = 0; i < 5; ++i){}
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} kernels loop
  // CHECK-NEXT: independent clause
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} i 'int'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 0
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: BinaryOperator{{.*}}'<'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'i' 'int'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 5
  // CHECK-NEXT: UnaryOperator{{.*}}++
  // CHECK-NEXT: DeclRefExpr{{.*}}'i' 'int'
  // CHECK-NEXT: CompoundStmt
}

template<typename T>
void TemplUses() {
  // CHECK: FunctionTemplateDecl{{.*}}TemplUses
  // CHECK-NEXT: TemplateTypeParmDecl
  // CHECK-NEXT: FunctionDecl{{.*}} TemplUses
  // CHECK-NEXT: CompoundStmt

#pragma acc parallel loop auto
  for(int i = 0; i < 5; ++i){}
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} parallel loop
  // CHECK-NEXT: auto clause
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} i 'int'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 0
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: BinaryOperator{{.*}}'<'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'i' 'int'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 5
  // CHECK-NEXT: UnaryOperator{{.*}}++
  // CHECK-NEXT: DeclRefExpr{{.*}}'i' 'int'
  // CHECK-NEXT: CompoundStmt

#pragma acc serial loop seq
  for(;;){}
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} serial loop
  // CHECK-NEXT: seq clause
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: CompoundStmt

#pragma acc kernels loop independent
  for(int i = 0; i < 5; ++i){}
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} kernels loop
  // CHECK-NEXT: independent clause
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} i 'int'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 0
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: BinaryOperator{{.*}}'<'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'i' 'int'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 5
  // CHECK-NEXT: UnaryOperator{{.*}}++
  // CHECK-NEXT: DeclRefExpr{{.*}}'i' 'int'
  // CHECK-NEXT: CompoundStmt

  // Instantiations.
  // CHECK-NEXT: FunctionDecl{{.*}}TemplUses 'void ()' implicit_instantiation
  // CHECK-NEXT: TemplateArgument
  // CHECK-NEXT: BuiltinType
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} parallel loop
  // CHECK-NEXT: auto clause
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} i 'int'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 0
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: BinaryOperator{{.*}}'<'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'i' 'int'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 5
  // CHECK-NEXT: UnaryOperator{{.*}}++
  // CHECK-NEXT: DeclRefExpr{{.*}}'i' 'int'
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} serial loop
  // CHECK-NEXT: seq clause
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} kernels loop
  // CHECK-NEXT: independent clause
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} i 'int'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 0
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: BinaryOperator{{.*}}'<'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'i' 'int'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 5
  // CHECK-NEXT: UnaryOperator{{.*}}++
  // CHECK-NEXT: DeclRefExpr{{.*}}'i' 'int'
  // CHECK-NEXT: CompoundStmt
}

void Inst() {
  TemplUses<int>();
}
#endif // PCH_HELPER
