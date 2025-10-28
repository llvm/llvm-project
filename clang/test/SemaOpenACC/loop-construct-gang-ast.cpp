// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -fopenacc -emit-pch -o %t %s
// RUN: %clang_cc1 %s -fopenacc -include-pch %t -ast-dump-all | FileCheck %s
#ifndef PCH_HELPER
#define PCH_HELPER

void NormalUses() {
  // CHECK: FunctionDecl{{.*}}NormalUses
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}<orphan>
  // CHECK-NEXT: gang clause dim
  // CHECK-NEXT: ConstantExpr{{.*}} 'int'
  // CHECK-NEXT: value: Int 1
  // CHECK-NEXT: IntegerLiteral{{.*}}'int' 1
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
#pragma acc loop gang(dim:1)
  for(int i = 0; i < 5; ++i);

  int Val;
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} used Val 'int'

  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}<orphan>
  // CHECK-NEXT: gang clause static
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'Val' 'int'
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
#pragma acc loop gang(static:Val)
  for(int i = 0; i < 5; ++i);

  // CHECK-NEXT: OpenACCComputeConstruct {{.*}} kernels
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: kernels
  // CHECK-NEXT: gang clause num
  // CHECK-NEXT: IntegerLiteral{{.*}}'int' 1
  // CHECK-NEXT: gang clause static
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'Val' 'int'
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
#pragma acc loop gang(num:1) gang(static:Val)
  for(int i = 0; i < 5; ++i);

  // CHECK-NEXT: OpenACCComputeConstruct {{.*}} parallel
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: parallel
  // CHECK-NEXT: gang clause dim static
  // CHECK-NEXT: ConstantExpr{{.*}} 'int'
  // CHECK-NEXT: value: Int 1
  // CHECK-NEXT: IntegerLiteral{{.*}}'int' 1
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'Val' 'int'
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
#pragma acc loop gang(dim:1, static:Val)
  for(int i = 0; i < 5; ++i);

  // CHECK-NEXT: OpenACCComputeConstruct {{.*}} serial
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: serial
  // CHECK-NEXT: gang clause static
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'Val' 'int'
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
#pragma acc loop gang(static:Val)
  for(int i = 0; i < 5; ++i);

  // CHECK-NEXT: OpenACCComputeConstruct {{.*}}serial
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: serial
  // CHECK-NEXT: gang clause static
  // CHECK-NEXT: OpenACCAsteriskSizeExpr
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
#pragma acc loop gang(static:*)
  for(int i = 0; i < 5; ++i);

  // CHECK-NEXT: OpenACCComputeConstruct {{.*}} serial
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: serial
  // CHECK-NEXT: gang clause
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
#pragma acc loop gang
  for(int i = 0; i < 5; ++i);
}

template<typename T, unsigned One>
void TemplateUses(T Val) {
  // CHECK: FunctionTemplateDecl{{.*}}TemplateUses
  // CHECK-NEXT: TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 T
  // CHECK-NEXT: NonTypeTemplateParmDecl{{.*}} referenced 'unsigned int' depth 0 index 1 One
  // CHECK-NEXT: FunctionDecl{{.*}} TemplateUses 'void (T)'
  // CHECK-NEXT: ParmVarDecl{{.*}} referenced Val 'T'
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}<orphan>
  // CHECK-NEXT: gang clause dim
  // CHECK-NEXT: DeclRefExpr{{.*}}'unsigned int' NonTypeTemplateParm{{.*}} 'One' 'unsigned int'
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
#pragma acc loop gang(dim:One)
  for(int i = 0; i < 5; ++i);

  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}<orphan>
  // CHECK-NEXT: gang clause static
  // CHECK-NEXT: DeclRefExpr{{.*}}'T' lvalue ParmVar{{.*}} 'Val' 'T'
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
#pragma acc loop gang(static:Val)
  for(int i = 0; i < 5; ++i);

  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}<orphan>
  // CHECK-NEXT: gang clause static
  // CHECK-NEXT: OpenACCAsteriskSizeExpr
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
#pragma acc loop gang(static:*)
  for(int i = 0; i < 5; ++i);

  // CHECK-NEXT: OpenACCComputeConstruct {{.*}} parallel
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: parallel
  // CHECK-NEXT: gang clause dim
  // CHECK-NEXT: DeclRefExpr{{.*}}'unsigned int' NonTypeTemplateParm{{.*}} 'One' 'unsigned int'
  // CHECK-NEXT: gang clause static
  // CHECK-NEXT: DeclRefExpr{{.*}}'T' lvalue ParmVar{{.*}} 'Val' 'T'
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
#pragma acc loop gang(dim:One) gang(static:Val)
  for(int i = 0; i < 5; ++i);

  // CHECK-NEXT: OpenACCComputeConstruct {{.*}} parallel
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: parallel
  // CHECK-NEXT: gang clause dim static
  // CHECK-NEXT: DeclRefExpr{{.*}}'unsigned int' NonTypeTemplateParm{{.*}} 'One' 'unsigned int'
  // CHECK-NEXT: DeclRefExpr{{.*}}'T' lvalue ParmVar{{.*}} 'Val' 'T'
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
#pragma acc loop gang(dim:One, static:Val)
  for(int i = 0; i < 5; ++i);

  // CHECK-NEXT: OpenACCComputeConstruct {{.*}} serial
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: serial
  // CHECK-NEXT: gang clause static
  // CHECK-NEXT: DeclRefExpr{{.*}}'T' lvalue ParmVar{{.*}} 'Val' 'T'
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
#pragma acc loop gang(static:Val)
  for(int i = 0; i < 5; ++i);

  // CHECK-NEXT: OpenACCComputeConstruct {{.*}} serial
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: serial
  // CHECK-NEXT: gang clause
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
#pragma acc loop gang
  for(int i = 0; i < 5; ++i);

  // Instantiation:
  // CHECK-NEXT: FunctionDecl{{.*}} used TemplateUses 'void (int)' implicit_instantiation
  // CHECK-NEXT: TemplateArgument type 'int'
  // CHECK-NEXT: BuiltinType{{.*}} 'int'
  // CHECK-NEXT: TemplateArgument integral '1U'
  // CHECK-NEXT: ParmVarDecl{{.*}} used Val 'int'
  // CHECK-NEXT: CompoundStmt
  //
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}<orphan>
  // CHECK-NEXT: gang clause dim
  // CHECK-NEXT: ConstantExpr{{.*}} 'unsigned int'
  // CHECK-NEXT: value: Int 1
  // CHECK-NEXT: SubstNonTypeTemplateParmExpr{{.*}}'unsigned int'
  // CHECK-NEXT: NonTypeTemplateParmDecl{{.*}} 'unsigned int' depth 0 index 1 One
  // CHECK-NEXT: IntegerLiteral{{.*}}'unsigned int' 1
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
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}<orphan>
  // CHECK-NEXT: gang clause static
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue ParmVar{{.*}} 'Val' 'int'
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
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}<orphan>
  // CHECK-NEXT: gang clause static
  // CHECK-NEXT: OpenACCAsteriskSizeExpr
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
  // CHECK-NEXT: gang clause dim
  // CHECK-NEXT: ConstantExpr{{.*}} 'unsigned int'
  // CHECK-NEXT: value: Int 1
  // CHECK-NEXT: SubstNonTypeTemplateParmExpr{{.*}}'unsigned int'
  // CHECK-NEXT: NonTypeTemplateParmDecl{{.*}} 'unsigned int' depth 0 index 1 One
  // CHECK-NEXT: IntegerLiteral{{.*}}'unsigned int' 1
  // CHECK-NEXT: gang clause static
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue ParmVar{{.*}} 'Val' 'int'
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
  // CHECK-NEXT: gang clause dim static
  // CHECK-NEXT: ConstantExpr{{.*}} 'unsigned int'
  // CHECK-NEXT: value: Int 1
  // CHECK-NEXT: SubstNonTypeTemplateParmExpr{{.*}}'unsigned int'
  // CHECK-NEXT: NonTypeTemplateParmDecl{{.*}} 'unsigned int' depth 0 index 1 One
  // CHECK-NEXT: IntegerLiteral{{.*}}'unsigned int' 1
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue ParmVar{{.*}} 'Val' 'int'
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
  // CHECK-NEXT: gang clause static
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue ParmVar{{.*}} 'Val' 'int'
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
  // CHECK-NEXT: gang clause
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

void inst() {
  TemplateUses<int, 1>(5);
}

#endif // PCH_HELPER
