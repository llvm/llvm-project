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
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: NullStmt
#pragma acc loop gang(dim:1)
  for(;;);

  int Val;
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} used Val 'int'

  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}<orphan>
  // CHECK-NEXT: gang clause static
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'Val' 'int'
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: NullStmt
#pragma acc loop gang(static:Val)
  for(;;);

  // CHECK-NEXT: OpenACCComputeConstruct 0x[[COMPUTE_ADDR:[0-9a-f]+]]{{.*}} kernels
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: 0x[[COMPUTE_ADDR]]
  // CHECK-NEXT: gang clause num
  // CHECK-NEXT: IntegerLiteral{{.*}}'int' 1
  // CHECK-NEXT: gang clause static
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'Val' 'int'
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: NullStmt
#pragma acc kernels
#pragma acc loop gang(num:1) gang(static:Val)
  for(;;);

  // CHECK-NEXT: OpenACCComputeConstruct 0x[[COMPUTE_ADDR:[0-9a-f]+]]{{.*}} parallel
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: 0x[[COMPUTE_ADDR]]
  // CHECK-NEXT: gang clause dim static
  // CHECK-NEXT: ConstantExpr{{.*}} 'int'
  // CHECK-NEXT: value: Int 1
  // CHECK-NEXT: IntegerLiteral{{.*}}'int' 1
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'Val' 'int'
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: NullStmt
#pragma acc parallel
#pragma acc loop gang(dim:1, static:Val)
  for(;;);

  // CHECK-NEXT: OpenACCComputeConstruct 0x[[COMPUTE_ADDR:[0-9a-f]+]]{{.*}} serial
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: 0x[[COMPUTE_ADDR]]
  // CHECK-NEXT: gang clause static
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'Val' 'int'
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: NullStmt
#pragma acc serial
#pragma acc loop gang(static:Val)
  for(;;);

  // CHECK-NEXT: OpenACCComputeConstruct 0x[[COMPUTE_ADDR:[0-9a-f]+]]{{.*}} serial
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: 0x[[COMPUTE_ADDR]]
  // CHECK-NEXT: gang clause static
  // CHECK-NEXT: OpenACCAsteriskSizeExpr
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: NullStmt
#pragma acc serial
#pragma acc loop gang(static:*)
  for(;;);

  // CHECK-NEXT: OpenACCComputeConstruct 0x[[COMPUTE_ADDR:[0-9a-f]+]]{{.*}} serial
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: 0x[[COMPUTE_ADDR]]
  // CHECK-NEXT: gang clause
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: NullStmt
#pragma acc serial
#pragma acc loop gang
  for(;;);
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
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: NullStmt
#pragma acc loop gang(dim:One)
  for(;;);

  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}<orphan>
  // CHECK-NEXT: gang clause static
  // CHECK-NEXT: DeclRefExpr{{.*}}'T' lvalue ParmVar{{.*}} 'Val' 'T'
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: NullStmt
#pragma acc loop gang(static:Val)
  for(;;);

  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}<orphan>
  // CHECK-NEXT: gang clause static
  // CHECK-NEXT: OpenACCAsteriskSizeExpr
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: NullStmt
#pragma acc loop gang(static:*)
  for(;;);

  // CHECK-NEXT: OpenACCComputeConstruct 0x[[COMPUTE_ADDR:[0-9a-f]+]]{{.*}} parallel
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: 0x[[COMPUTE_ADDR]]
  // CHECK-NEXT: gang clause dim
  // CHECK-NEXT: DeclRefExpr{{.*}}'unsigned int' NonTypeTemplateParm{{.*}} 'One' 'unsigned int'
  // CHECK-NEXT: gang clause static
  // CHECK-NEXT: DeclRefExpr{{.*}}'T' lvalue ParmVar{{.*}} 'Val' 'T'
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: NullStmt
#pragma acc parallel
#pragma acc loop gang(dim:One) gang(static:Val)
  for(;;);

  // CHECK-NEXT: OpenACCComputeConstruct 0x[[COMPUTE_ADDR:[0-9a-f]+]]{{.*}} parallel
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: 0x[[COMPUTE_ADDR]]
  // CHECK-NEXT: gang clause dim static
  // CHECK-NEXT: DeclRefExpr{{.*}}'unsigned int' NonTypeTemplateParm{{.*}} 'One' 'unsigned int'
  // CHECK-NEXT: DeclRefExpr{{.*}}'T' lvalue ParmVar{{.*}} 'Val' 'T'
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: NullStmt
#pragma acc parallel
#pragma acc loop gang(dim:One, static:Val)
  for(;;);

  // CHECK-NEXT: OpenACCComputeConstruct 0x[[COMPUTE_ADDR:[0-9a-f]+]]{{.*}} serial
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: 0x[[COMPUTE_ADDR]]
  // CHECK-NEXT: gang clause static
  // CHECK-NEXT: DeclRefExpr{{.*}}'T' lvalue ParmVar{{.*}} 'Val' 'T'
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: NullStmt
#pragma acc serial
#pragma acc loop gang(static:Val)
  for(;;);

  // CHECK-NEXT: OpenACCComputeConstruct 0x[[COMPUTE_ADDR:[0-9a-f]+]]{{.*}} serial
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: 0x[[COMPUTE_ADDR]]
  // CHECK-NEXT: gang clause
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: NullStmt
#pragma acc serial
#pragma acc loop gang
  for(;;);

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
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: NullStmt
  //
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}<orphan>
  // CHECK-NEXT: gang clause static
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue ParmVar{{.*}} 'Val' 'int'
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: NullStmt
  //
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}<orphan>
  // CHECK-NEXT: gang clause static
  // CHECK-NEXT: OpenACCAsteriskSizeExpr
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: NullStmt
  //
  // CHECK-NEXT: OpenACCComputeConstruct 0x[[COMPUTE_ADDR:[0-9a-f]+]]{{.*}} parallel
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: 0x[[COMPUTE_ADDR]]
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
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: NullStmt
  //
  // CHECK-NEXT: OpenACCComputeConstruct 0x[[COMPUTE_ADDR:[0-9a-f]+]]{{.*}} parallel
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: 0x[[COMPUTE_ADDR]]
  // CHECK-NEXT: gang clause dim static
  // CHECK-NEXT: ConstantExpr{{.*}} 'unsigned int'
  // CHECK-NEXT: value: Int 1
  // CHECK-NEXT: SubstNonTypeTemplateParmExpr{{.*}}'unsigned int'
  // CHECK-NEXT: NonTypeTemplateParmDecl{{.*}} 'unsigned int' depth 0 index 1 One
  // CHECK-NEXT: IntegerLiteral{{.*}}'unsigned int' 1
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue ParmVar{{.*}} 'Val' 'int'
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: NullStmt
  //
  // CHECK-NEXT: OpenACCComputeConstruct 0x[[COMPUTE_ADDR:[0-9a-f]+]]{{.*}} serial
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: 0x[[COMPUTE_ADDR]]
  // CHECK-NEXT: gang clause static
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue ParmVar{{.*}} 'Val' 'int'
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: NullStmt
  //
  // CHECK-NEXT: OpenACCComputeConstruct 0x[[COMPUTE_ADDR:[0-9a-f]+]]{{.*}} serial
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: 0x[[COMPUTE_ADDR]]
  // CHECK-NEXT: gang clause
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: NullStmt
}

void inst() {
  TemplateUses<int, 1>(5);
}

#endif // PCH_HELPER
