// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -fopenacc -emit-pch -o %t %s
// RUN: %clang_cc1 %s -fopenacc -include-pch %t -ast-dump-all | FileCheck %s
#ifndef PCH_HELPER
#define PCH_HELPER

template<unsigned I, typename ConvertsToInt, typename Int>
void TemplUses(ConvertsToInt CTI, Int IsI) {
  // CHECK: FunctionTemplateDecl{{.*}}TemplUses
  // CHECK-NEXT: NonTypeTemplateParmDecl{{.*}} 'unsigned int' depth 0 index 0 I
  // CHECK-NEXT: TemplateTypeParmDecl{{.*}}typename depth 0 index 1 ConvertsToInt
  // CHECK-NEXT: TemplateTypeParmDecl{{.*}}typename depth 0 index 2 Int
  // CHECK-NEXT: FunctionDecl{{.*}}TemplUses 'void (ConvertsToInt, Int)'
  // CHECK-NEXT: ParmVarDecl{{.*}}CTI 'ConvertsToInt'
  // CHECK-NEXT: ParmVarDecl{{.*}}IsI 'Int'
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}<orphan>
  // CHECK-NEXT: worker clause{{.*}}
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
  // CHECK-NEXT: NullStmt
#pragma acc loop worker
  for(int i = 0; i < 5; ++i);

  // CHECK-NEXT: OpenACCComputeConstruct {{.*}} parallel
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: parallel
  // CHECK-NEXT: worker clause{{.*}}
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
  // CHECK-NEXT: NullStmt
#pragma acc parallel
#pragma acc loop worker
  for(int i = 0; i < 5; ++i);

  // CHECK-NEXT: OpenACCComputeConstruct {{.*}} serial
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: serial
  // CHECK-NEXT: worker clause{{.*}}
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
  // CHECK-NEXT: NullStmt
#pragma acc serial
#pragma acc loop worker
  for(int i = 0; i < 5; ++i);

  // CHECK-NEXT: OpenACCComputeConstruct {{.*}} kernels
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: kernels
  // CHECK-NEXT: worker clause{{.*}}
  // CHECK-NEXT: DeclRefExpr{{.*}} 'ConvertsToInt' lvalue ParmVar
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
  // CHECK-NEXT: NullStmt
#pragma acc kernels
#pragma acc loop worker(CTI)
  for(int i = 0; i < 5; ++i);

  // CHECK-NEXT: OpenACCComputeConstruct {{.*}} kernels
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: kernels
  // CHECK-NEXT: worker clause{{.*}}
  // CHECK-NEXT: DeclRefExpr{{.*}} 'Int' lvalue ParmVar
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
  // CHECK-NEXT: NullStmt
#pragma acc kernels
#pragma acc loop worker(num:IsI)
  for(int i = 0; i < 5; ++i);

  // CHECK-NEXT: OpenACCComputeConstruct {{.*}} kernels
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: kernels
  // CHECK-NEXT: worker clause{{.*}}
  // CHECK-NEXT: DeclRefExpr{{.*}} 'unsigned int' NonTypeTemplateParm{{.*}}'I' 'unsigned int'
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
  // CHECK-NEXT: NullStmt
#pragma acc kernels
#pragma acc loop worker(num:I)
  for(int i = 0; i < 5; ++i);

  // Instantiations:
  // CHECK-NEXT: FunctionDecl{{.*}} TemplUses 'void (Converts, int)' implicit_instantiation
  // CHECK-NEXT: TemplateArgument integral '3U'
  // CHECK-NEXT: TemplateArgument type 'Converts'
  // CHECK-NEXT: RecordType{{.*}}'Converts'
  // CHECK-NEXT: CXXRecord{{.*}}'Converts
  // CHECK-NEXT: TemplateArgument type 'int'
  // CHECK-NEXT: BuiltinType{{.*}}'int'
  // CHECK-NEXT: ParmVarDecl{{.*}} CTI 'Converts'
  // CHECK-NEXT: ParmVarDecl{{.*}} IsI 'int'
  // CHECK-NEXT: CompoundStmt
  //
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}<orphan>
  // CHECK-NEXT: worker clause{{.*}}
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
  // CHECK-NEXT: NullStmt
  //
  // CHECK-NEXT: OpenACCComputeConstruct {{.*}} parallel
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: parallel
  // CHECK-NEXT: worker clause{{.*}}
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
  // CHECK-NEXT: NullStmt
  //
  // CHECK-NEXT: OpenACCComputeConstruct {{.*}} serial
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: serial
  // CHECK-NEXT: worker clause{{.*}}
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
  // CHECK-NEXT: NullStmt
  //
  // CHECK-NEXT: OpenACCComputeConstruct {{.*}} kernels
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: kernels
  // CHECK-NEXT: worker clause{{.*}}
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}} 'int'
  // CHECK-NEXT: MemberExpr{{.*}} .operator int
  // CHECK-NEXT: DeclRefExpr{{.*}} 'Converts' lvalue ParmVar
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
  // CHECK-NEXT: NullStmt
  //
  // CHECK-NEXT: OpenACCComputeConstruct {{.*}} kernels
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: kernels
  // CHECK-NEXT: worker clause{{.*}}
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int' lvalue ParmVar{{.*}} 'IsI' 'int'
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
  // CHECK-NEXT: NullStmt
  //
  // CHECK-NEXT: OpenACCComputeConstruct {{.*}} kernels
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: kernels
  // CHECK-NEXT: worker clause{{.*}}
  // CHECK-NEXT: SubstNonTypeTemplateParmExpr{{.*}}'unsigned int'
  // CHECK-NEXT: NonTypeTemplateParmDecl{{.*}}'unsigned int' depth 0 index 0 I
  // CHECK-NEXT: IntegerLiteral{{.*}} 'unsigned int' 3
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
  // CHECK-NEXT: NullStmt
}

struct Converts{
  operator int();
};

void uses() {
  // CHECK: FunctionDecl{{.*}} uses
  // CHECK-NEXT: CompoundStmt
  //
  // CHECK-NEXT: CallExpr
  TemplUses<3>(Converts{}, 5);

  // CHECK: OpenACCLoopConstruct{{.*}}<orphan>
  // CHECK-NEXT: worker clause{{.*}}
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
  // CHECK-NEXT: NullStmt
#pragma acc loop worker
  for(int i = 0; i < 5; ++i);

  // CHECK-NEXT: OpenACCComputeConstruct {{.*}} parallel
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: parallel
  // CHECK-NEXT: worker clause{{.*}}
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
  // CHECK-NEXT: NullStmt
#pragma acc parallel
#pragma acc loop worker
  for(int i = 0; i < 5; ++i);

  // CHECK-NEXT: OpenACCComputeConstruct {{.*}} serial
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: serial
  // CHECK-NEXT: worker clause{{.*}}
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
  // CHECK-NEXT: NullStmt
#pragma acc serial
#pragma acc loop worker
  for(int i = 0; i < 5; ++i);

  Converts CTI;
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl
  // CHECK-NEXT: CXXConstructExpr

  // CHECK-NEXT: OpenACCComputeConstruct {{.*}} kernels
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: kernels
  // CHECK-NEXT: worker clause{{.*}}
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}} 'int'
  // CHECK-NEXT: MemberExpr{{.*}} .operator int
  // CHECK-NEXT: DeclRefExpr{{.*}}'Converts' lvalue Var
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
  // CHECK-NEXT: NullStmt
#pragma acc kernels
#pragma acc loop worker(CTI)
  for(int i = 0; i < 5; ++i);

  int IsI;
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl

  // CHECK-NEXT: OpenACCComputeConstruct {{.*}} kernels
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: kernels
  // CHECK-NEXT: worker clause{{.*}}
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue Var
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
  // CHECK-NEXT: NullStmt
#pragma acc kernels
#pragma acc loop worker(num:IsI)
  for(int i = 0; i < 5; ++i);
}

#endif // PCH_HELPER
