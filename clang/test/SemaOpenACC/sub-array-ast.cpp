// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -fopenacc -emit-pch -o %t %s
// RUN: %clang_cc1 %s -fopenacc -include-pch %t -ast-dump-all | FileCheck %s

#ifndef PCH_HELPER
#define PCH_HELPER

constexpr int returns_3() { return 3; }

void Func(int i, int j) {
  // CHECK: FunctionDecl{{.*}}Func
  // CHECK-NEXT: ParmVarDecl{{.*}} i 'int'
  // CHECK-NEXT: ParmVarDecl{{.*}} j 'int'
  // CHECK-NEXT: CompoundStmt
  int array[5];
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} array 'int[5]'
  int VLA[i];
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} VLA 'int[i]'
  int *ptr;
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} ptr 'int *'

#pragma acc parallel private(array[returns_3():])
  while (true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int *' <ArrayToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int[5]' lvalue Var{{.*}} 'array' 'int[5]'
  // CHECK-NEXT: CallExpr{{.*}} 'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int ()' lvalue Function{{.*}} 'returns_3' 'int ()'
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel private(array[:1])
  while (true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int *' <ArrayToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int[5]' lvalue Var{{.*}} 'array' 'int[5]'
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 1
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel private(array[returns_3():1])
  while (true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int *' <ArrayToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int[5]' lvalue Var{{.*}} 'array' 'int[5]'
  // CHECK-NEXT: CallExpr{{.*}} 'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int ()' lvalue Function{{.*}} 'returns_3' 'int ()'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 1
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel private(array[i:j])
  while (true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int *' <ArrayToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int[5]' lvalue Var{{.*}} 'array' 'int[5]'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int' lvalue ParmVar{{.*}} 'i' 'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int' lvalue ParmVar{{.*}} 'j' 'int'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel private(VLA[:1])
  while (true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int *' <ArrayToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int[i]' lvalue Var{{.*}} 'VLA' 'int[i]'
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 1
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel private(VLA[returns_3():1])
  while (true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int *' <ArrayToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int[i]' lvalue Var{{.*}} 'VLA' 'int[i]'
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int ()' lvalue Function{{.*}} 'returns_3' 'int ()'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 1
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel private(VLA[i:j])
  while (true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int *' <ArrayToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int[i]' lvalue Var{{.*}} 'VLA' 'int[i]'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int' lvalue ParmVar{{.*}} 'i' 'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int' lvalue ParmVar{{.*}} 'j' 'int'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel private(ptr[:1])
  while (true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int *' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int *' lvalue Var{{.*}} 'ptr' 'int *'
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 1
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel private(ptr[returns_3():1])
  while (true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int *' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int *' lvalue Var{{.*}} 'ptr' 'int *'
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int ()' lvalue Function{{.*}} 'returns_3' 'int ()'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 1
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel private(ptr[i:j])
  while (true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int *' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int *' lvalue Var{{.*}} 'ptr' 'int *'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int' lvalue ParmVar{{.*}} 'i' 'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int' lvalue ParmVar{{.*}} 'j' 'int'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt
}

template<typename T, unsigned I, auto &CEArray>
void Templ(int i){
  // CHECK-NEXT: FunctionTemplateDecl{{.*}}Templ
  // CHECK-NEXT: TemplateTypeParmDecl{{.*}} typename depth 0 index 0 T
  // CHECK-NEXT: NonTypeTemplateParmDecl{{.*}} 'unsigned int' depth 0 index 1 I
  // CHECK-NEXT: NonTypeTemplateParmDecl{{.*}} 'auto &' depth 0 index 2 CEArray
  // CHECK-NEXT: FunctionDecl{{.*}}Templ 'void (int)'
  // CHECK-NEXT: ParmVarDecl{{.*}} i 'int'
  // CHECK-NEXT: CompoundStmt
  T array[I+2];
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} array 'T[I + 2]'
  T VLA[i];
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} VLA 'T[i]'
  T *ptr;
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} ptr 'T *'

#pragma acc parallel private(array[returns_3():])
  while (true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: DeclRefExpr{{.*}} 'T[I + 2]' lvalue Var{{.*}} 'array' 'T[I + 2]'
  // CHECK-NEXT: CallExpr{{.*}} 'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int ()' lvalue Function{{.*}}'returns_3' 'int ()'
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel private(array[:I])
  while (true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: DeclRefExpr{{.*}} 'T[I + 2]' lvalue Var{{.*}} 'array' 'T[I + 2]'
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'unsigned int' NonTypeTemplateParm{{.*}} 'I' 'unsigned int'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel private(array[returns_3()-2:I])
  while (true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: DeclRefExpr{{.*}} 'T[I + 2]' lvalue Var{{.*}} 'array' 'T[I + 2]'
  // CHECK-NEXT: BinaryOperator{{.*}} 'int' '-'
  // CHECK-NEXT: CallExpr{{.*}} 'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int ()' lvalue Function{{.*}} 'returns_3' 'int ()'
  // CHECK-NEXT: IntegerLiteral{{.*}} 2
  // CHECK-NEXT: DeclRefExpr{{.*}} 'unsigned int' NonTypeTemplateParm{{.*}} 'I' 'unsigned int'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel private(array[i:i])
  while (true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: DeclRefExpr{{.*}} 'T[I + 2]' lvalue Var{{.*}} 'array' 'T[I + 2]'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int' lvalue ParmVar{{.*}} 'i' 'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int' lvalue ParmVar{{.*}} 'i' 'int'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel private(VLA[:I])
  while (true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: DeclRefExpr{{.*}} 'T[i]' lvalue Var{{.*}} 'VLA' 'T[i]'
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'unsigned int' NonTypeTemplateParm{{.*}} 'I' 'unsigned int'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel private(VLA[returns_3():I])
  while (true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: DeclRefExpr{{.*}} 'T[i]' lvalue Var{{.*}} 'VLA' 'T[i]'
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int ()' lvalue Function{{.*}} 'returns_3' 'int ()'
  // CHECK-NEXT: DeclRefExpr{{.*}} 'unsigned int' NonTypeTemplateParm{{.*}} 'I' 'unsigned int'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel private(VLA[i:i])
  while (true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: DeclRefExpr{{.*}} 'T[i]' lvalue Var{{.*}} 'VLA' 'T[i]'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int' lvalue ParmVar{{.*}} 'i' 'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int' lvalue ParmVar{{.*}} 'i' 'int'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel private(ptr[:I])
  while (true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: DeclRefExpr{{.*}} 'T *' lvalue Var{{.*}} 'ptr' 'T *'
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'unsigned int' NonTypeTemplateParm{{.*}} 'I' 'unsigned int'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel private(ptr[returns_3():I])
  while (true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: DeclRefExpr{{.*}} 'T *' lvalue Var{{.*}} 'ptr' 'T *'
  // CHECK-NEXT: CallExpr{{.*}} 'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int ()' lvalue Function{{.*}} 'returns_3' 'int ()'
  // CHECK-NEXT: DeclRefExpr{{.*}} 'unsigned int' NonTypeTemplateParm{{.*}} 'I' 'unsigned int'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel private(ptr[i:i])
  while (true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: DeclRefExpr{{.*}} 'T *' lvalue Var{{.*}} 'ptr' 'T *'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int' lvalue ParmVar{{.*}} 'i' 'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int' lvalue ParmVar{{.*}} 'i' 'int'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel private(CEArray[returns_3() - 2: I])
  while (true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: DeclRefExpr{{.*}} 'auto' lvalue NonTypeTemplateParm{{.*}} 'CEArray' 'auto &'
  // CHECK-NEXT: BinaryOperator{{.*}} 'int' '-'
  // CHECK-NEXT: CallExpr{{.*}} 'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int ()' lvalue Function{{.*}} 'returns_3' 'int ()'
  // CHECK-NEXT: IntegerLiteral{{.*}} 2
  // CHECK-NEXT: DeclRefExpr{{.*}} 'unsigned int' NonTypeTemplateParm{{.*}} 'I' 'unsigned int'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel private(CEArray[: I])
  while (true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: DeclRefExpr{{.*}} 'auto' lvalue NonTypeTemplateParm{{.*}} 'CEArray' 'auto &'
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'unsigned int' NonTypeTemplateParm{{.*}} 'I' 'unsigned int'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt


  // Instantiation:
  // CHECK-NEXT: FunctionDecl{{.*}} Templ 'void (int)' implicit_instantiation
  // CHECK-NEXT: TemplateArgument{{.*}} 'int'
  // CHECK-NEXT: BuiltinType{{.*}} 'int'
  // CHECK-NEXT: TemplateArgument integral '3U'
  // CHECK-NEXT: TemplateArgument decl
  // CHECK-NEXT: Var{{.*}} 'CEArray' 'const int[5]'
  // CHECK-NEXT: ParmVarDecl{{.*}} i 'int'
  // CHECK-NEXT: CompoundStmt

  // T array[I+2];
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} array 'int[5]'
  // T VLA[i];
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} VLA 'int[i]'
  // T *ptr;
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} ptr 'int *'

//#pragma acc parallel private(array[returns_3():])
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int *' <ArrayToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int[5]' lvalue Var{{.*}} 'array' 'int[5]'
  // CHECK-NEXT: CallExpr{{.*}} 'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int ()' lvalue Function{{.*}}'returns_3' 'int ()'
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

//#pragma acc parallel private(array[:I])
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int *' <ArrayToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int[5]' lvalue Var{{.*}} 'array' 'int[5]'
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: SubstNonTypeTemplateParmExpr{{.*}} 'unsigned int'
  // CHECK-NEXT: NonTypeTemplateParmDecl{{.*}} 'unsigned int' depth 0 index 1 I
  // CHECK-NEXT: IntegerLiteral{{.*}} 'unsigned int' 3
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

//#pragma acc parallel private(array[returns_3()-2:I])
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int *' <ArrayToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int[5]' lvalue Var{{.*}} 'array' 'int[5]'
  // CHECK-NEXT: BinaryOperator{{.*}} 'int' '-'
  // CHECK-NEXT: CallExpr{{.*}} 'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int ()' lvalue Function{{.*}} 'returns_3' 'int ()'
  // CHECK-NEXT: IntegerLiteral{{.*}} 2
  // CHECK-NEXT: SubstNonTypeTemplateParmExpr{{.*}} 'unsigned int'
  // CHECK-NEXT: NonTypeTemplateParmDecl{{.*}} 'unsigned int' depth 0 index 1 I
  // CHECK-NEXT: IntegerLiteral{{.*}} 'unsigned int' 3
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

//#pragma acc parallel private(array[i:i])
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int *' <ArrayToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int[5]' lvalue Var{{.*}} 'array' 'int[5]'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int' lvalue ParmVar{{.*}} 'i' 'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int' lvalue ParmVar{{.*}} 'i' 'int'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

//#pragma acc parallel private(VLA[:I])
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int *' <ArrayToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int[i]' lvalue Var{{.*}} 'VLA' 'int[i]'
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: SubstNonTypeTemplateParmExpr{{.*}} 'unsigned int'
  // CHECK-NEXT: NonTypeTemplateParmDecl{{.*}} 'unsigned int' depth 0 index 1 I
  // CHECK-NEXT: IntegerLiteral{{.*}} 'unsigned int' 3
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

//#pragma acc parallel private(VLA[returns_3():I])
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int *' <ArrayToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int[i]' lvalue Var{{.*}} 'VLA' 'int[i]'
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int ()' lvalue Function{{.*}} 'returns_3' 'int ()'
  // CHECK-NEXT: SubstNonTypeTemplateParmExpr{{.*}} 'unsigned int'
  // CHECK-NEXT: NonTypeTemplateParmDecl{{.*}} 'unsigned int' depth 0 index 1 I
  // CHECK-NEXT: IntegerLiteral{{.*}} 'unsigned int' 3
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

//#pragma acc parallel private(VLA[i:i])
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int *' <ArrayToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int[i]' lvalue Var{{.*}} 'VLA' 'int[i]'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int' lvalue ParmVar{{.*}} 'i' 'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int' lvalue ParmVar{{.*}} 'i' 'int'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

//#pragma acc parallel private(ptr[:I])
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int *' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int *' lvalue Var{{.*}} 'ptr' 'int *'
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: SubstNonTypeTemplateParmExpr{{.*}} 'unsigned int'
  // CHECK-NEXT: NonTypeTemplateParmDecl{{.*}} 'unsigned int' depth 0 index 1 I
  // CHECK-NEXT: IntegerLiteral{{.*}} 'unsigned int' 3
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

//#pragma acc parallel private(ptr[returns_3():I])
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int *' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int *' lvalue Var{{.*}} 'ptr' 'int *'
  // CHECK-NEXT: CallExpr{{.*}} 'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int ()' lvalue Function{{.*}} 'returns_3' 'int ()'
  // CHECK-NEXT: SubstNonTypeTemplateParmExpr{{.*}} 'unsigned int'
  // CHECK-NEXT: NonTypeTemplateParmDecl{{.*}} 'unsigned int' depth 0 index 1 I
  // CHECK-NEXT: IntegerLiteral{{.*}} 'unsigned int' 3
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

//#pragma acc parallel private(ptr[i:i])
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int *' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int *' lvalue Var{{.*}} 'ptr' 'int *'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int' lvalue ParmVar{{.*}} 'i' 'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int' lvalue ParmVar{{.*}} 'i' 'int'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

//#pragma acc parallel private(CEArray[returns_3() - 2: I])
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'const int *' <ArrayToPointerDecay>
  // CHECK-NEXT: SubstNonTypeTemplateParmExpr{{.*}} 'const int[5]' lvalue
  // CHECK-NEXT: NonTypeTemplateParmDecl{{.*}} 'auto &' depth 0 index 2 CEArray
  // CHECK-NEXT: DeclRefExpr{{.*}} 'const int[5]' lvalue Var{{.*}}'CEArray' 'const int[5]'
  // CHECK-NEXT: BinaryOperator{{.*}} 'int' '-'
  // CHECK-NEXT: CallExpr{{.*}} 'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int ()' lvalue Function{{.*}} 'returns_3' 'int ()'
  // CHECK-NEXT: IntegerLiteral{{.*}} 2
  // CHECK-NEXT: SubstNonTypeTemplateParmExpr{{.*}} 'unsigned int'
  // CHECK-NEXT: NonTypeTemplateParmDecl{{.*}} 'unsigned int' depth 0 index 1 I
  // CHECK-NEXT: IntegerLiteral{{.*}} 'unsigned int' 3
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

//#pragma acc parallel private(CEArray[: I])
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'const int *' <ArrayToPointerDecay>
  // CHECK-NEXT: SubstNonTypeTemplateParmExpr{{.*}} 'const int[5]' lvalue
  // CHECK-NEXT: NonTypeTemplateParmDecl{{.*}} 'auto &' depth 0 index 2 CEArray
  // CHECK-NEXT: DeclRefExpr{{.*}} 'const int[5]' lvalue Var{{.*}}'CEArray' 'const int[5]'
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: SubstNonTypeTemplateParmExpr{{.*}} 'unsigned int'
  // CHECK-NEXT: NonTypeTemplateParmDecl{{.*}} 'unsigned int' depth 0 index 1 I
  // CHECK-NEXT: IntegerLiteral{{.*}} 'unsigned int' 3
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt
}

// CHECK-NEXT: FunctionDecl{{.*}}inst
void inst() {
  static constexpr int CEArray[5]={1,2,3,4,5};
  Templ<int, 3, CEArray>(5);
}
#endif
