// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -fopenacc -emit-pch -o %t %s
// RUN: %clang_cc1 %s -fopenacc -include-pch %t -ast-dump-all | FileCheck %s
#ifndef PCH_HELPER
#define PCH_HELPER

int Global;
short GlobalArray[5];

void NormalUses(float *PointerParam) {
  // CHECK: FunctionDecl{{.*}}NormalUses
  // CHECK: ParmVarDecl
  // CHECK-NEXT: CompoundStmt

#pragma acc parallel loop private(Global, GlobalArray[2])
  for (int i = 5; i < 10; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} parallel loop
  // CHECK-NEXT: private clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue Var{{.*}}'Global' 'int'
  // CHECK-NEXT: ArraySubscriptExpr{{.*}}'short' lvalue
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'short *' <ArrayToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'short[5]' lvalue Var{{.*}}'GlobalArray' 'short[5]'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 2
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc serial loop private(GlobalArray, PointerParam[Global])
  for (int i = 5; i < 10; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} serial loop
  // CHECK-NEXT: private clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'short[5]' lvalue Var{{.*}}'GlobalArray' 'short[5]'
  // CHECK-NEXT: ArraySubscriptExpr{{.*}}'float' lvalue
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'float *' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}}'float *' lvalue ParmVar{{.*}}'PointerParam' 'float *'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue Var{{.*}}'Global' 'int'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc parallel loop firstprivate(GlobalArray, PointerParam[Global])
  for (int i = 5; i < 10; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} parallel loop
  // CHECK-NEXT: firstprivate clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'short[5]' lvalue Var{{.*}}'GlobalArray' 'short[5]'
  // CHECK-NEXT: ArraySubscriptExpr{{.*}}'float' lvalue
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'float *' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}}'float *' lvalue ParmVar{{.*}}'PointerParam' 'float *'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue Var{{.*}}'Global' 'int'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc serial loop firstprivate(GlobalArray, PointerParam[Global : Global])
  for (int i = 5; i < 10; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} serial loop
  // CHECK-NEXT: firstprivate clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'short[5]' lvalue Var{{.*}}'GlobalArray' 'short[5]'
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'float *' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}}'float *' lvalue ParmVar{{.*}} 'PointerParam' 'float *'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue Var{{.*}}'Global' 'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue Var{{.*}}'Global' 'int'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

}
template<auto &NTTP, typename T, typename U>
void TemplUses(T t, U u, T*PointerParam) {

  // CHECK: FunctionTemplateDecl{{.*}}TemplUses
  // CHECK-NEXT: NonTypeTemplateParmDecl {{.*}}referenced 'auto &' depth 0 index 0 NTTP
  // CHECK-NEXT: TemplateTypeParmDecl{{.*}}typename depth 0 index 1 T
  // CHECK-NEXT: TemplateTypeParmDecl{{.*}}typename depth 0 index 2 U
  // CHECK-NEXT: FunctionDecl{{.*}} TemplUses 'void (T, U, T *)'
  // CHECK-NEXT: ParmVarDecl{{.*}} referenced t 'T'
  // CHECK-NEXT: ParmVarDecl{{.*}} referenced u 'U'
  // CHECK-NEXT: ParmVarDecl{{.*}} referenced PointerParam 'T *'
  // CHECK-NEXT: CompoundStmt
#pragma acc serial loop private(GlobalArray, PointerParam[Global])
  for (int i = 5; i < 10; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} serial loop
  // CHECK-NEXT: private clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'short[5]' lvalue Var{{.*}}'GlobalArray' 'short[5]'
  // CHECK-NEXT: ArraySubscriptExpr{{.*}}'T' lvalue
  // CHECK-NEXT: DeclRefExpr{{.*}}'T *' lvalue ParmVar{{.*}}'PointerParam' 'T *'
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue Var{{.*}}'Global' 'int'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc kernels loop private(t, u)
  for (int i = 5; i < 10; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} kernels loop
  // CHECK-NEXT: private clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'T' lvalue ParmVar{{.*}} 't' 'T'
  // CHECK-NEXT: DeclRefExpr{{.*}}'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc parallel loop firstprivate(t, u)
  for (int i = 5; i < 10; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} parallel loop
  // CHECK-NEXT: firstprivate clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'T' lvalue ParmVar{{.*}} 't' 'T'
  // CHECK-NEXT: DeclRefExpr{{.*}}'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc kernels loop private(t) private(u)
  for (int i = 5; i < 10; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} kernels loop
  // CHECK-NEXT: private clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'T' lvalue ParmVar{{.*}} 't' 'T'
  // CHECK-NEXT: private clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc kernels loop private(t) private(NTTP, u)
  for (int i = 5; i < 10; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} kernels loop
  // CHECK-NEXT: private clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'T' lvalue ParmVar{{.*}} 't' 'T'
  // CHECK-NEXT: private clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'auto' lvalue NonTypeTemplateParm{{.*}} 'NTTP' 'auto &'
  // CHECK-NEXT: DeclRefExpr{{.*}}'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc parallel loop private(t) firstprivate(NTTP, u)
  for (int i = 5; i < 10; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} parallel loop
  // CHECK-NEXT: private clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'T' lvalue ParmVar{{.*}} 't' 'T'
  // CHECK-NEXT: firstprivate clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'auto' lvalue NonTypeTemplateParm{{.*}} 'NTTP' 'auto &'
  // CHECK-NEXT: DeclRefExpr{{.*}}'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

  // Check the instantiated versions of the above.
  // CHECK-NEXT: FunctionDecl{{.*}} used TemplUses 'void (int, int *, int *)' implicit_instantiation
  // CHECK-NEXT: TemplateArgument decl
  // CHECK-NEXT: Var{{.*}} 'CEVar' 'const unsigned int'
  // CHECK-NEXT: TemplateArgument type 'int'
  // CHECK-NEXT: BuiltinType{{.*}} 'int'
  // CHECK-NEXT: TemplateArgument type 'int[1]'
  // CHECK-NEXT: ConstantArrayType{{.*}} 'int[1]'{{.*}} 1
  // CHECK-NEXT: BuiltinType{{.*}} 'int'
  // CHECK-NEXT: ParmVarDecl{{.*}} used t 'int'
  // CHECK-NEXT: ParmVarDecl{{.*}} used u 'int *'
  // CHECK-NEXT: ParmVarDecl{{.*}} used PointerParam 'int *'
  // CHECK-NEXT: CompoundStmt

// #pragma acc serial loop private(GlobalArray, PointerParam[Global])
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} serial loop
  // CHECK-NEXT: private clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'short[5]' lvalue Var{{.*}}'GlobalArray' 'short[5]'
  // CHECK-NEXT: ArraySubscriptExpr{{.*}}'int' lvalue
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int *' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int *' lvalue ParmVar{{.*}}'PointerParam' 'int *'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue Var{{.*}}'Global' 'int'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

// #pragma acc kernels loop private(t, u)
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} kernels loop
  // CHECK-NEXT: private clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue ParmVar{{.*}} 't' 'int'
  // CHECK-NEXT: DeclRefExpr{{.*}}'int *' lvalue ParmVar{{.*}} 'u' 'int *'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

// #pragma acc parallel loop firstprivate(t, u)
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} parallel loop
  // CHECK-NEXT: firstprivate clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue ParmVar{{.*}} 't' 'int'
  // CHECK-NEXT: DeclRefExpr{{.*}}'int *' lvalue ParmVar{{.*}} 'u' 'int *'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

// #pragma acc kernels loop private(t) private(u)
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} kernels loop
  // CHECK-NEXT: private clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue ParmVar{{.*}} 't' 'int'
  // CHECK-NEXT: private clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'int *' lvalue ParmVar{{.*}} 'u' 'int *'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

// #pragma acc kernels loop private(t) private(NTTP, u)
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} kernels loop
  // CHECK-NEXT: private clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue ParmVar{{.*}} 't' 'int'
  // CHECK-NEXT: private clause
  // CHECK-NEXT: SubstNonTypeTemplateParmExpr{{.*}}'const unsigned int' lvalue
  // CHECK-NEXT: NonTypeTemplateParmDecl{{.*}} referenced 'auto &' depth 0 index 0 NTTP
  // CHECK-NEXT: DeclRefExpr{{.*}}'const unsigned int' lvalue Var{{.*}} 'CEVar' 'const unsigned int'
  // CHECK-NEXT: DeclRefExpr{{.*}}'int *' lvalue ParmVar{{.*}} 'u' 'int *'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

// #pragma acc parallel loop private(t) firstprivate(NTTP, u)
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} parallel loop
  // CHECK-NEXT: private clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue ParmVar{{.*}} 't' 'int'
  // CHECK-NEXT: firstprivate clause
  // CHECK-NEXT: SubstNonTypeTemplateParmExpr{{.*}}'const unsigned int' lvalue
  // CHECK-NEXT: NonTypeTemplateParmDecl{{.*}} referenced 'auto &' depth 0 index 0 NTTP
  // CHECK-NEXT: DeclRefExpr{{.*}}'const unsigned int' lvalue Var{{.*}} 'CEVar' 'const unsigned int'
  // CHECK-NEXT: DeclRefExpr{{.*}}'int *' lvalue ParmVar{{.*}} 'u' 'int *'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt
}

void Inst() {
  static constexpr unsigned CEVar = 1;
  int i;
  int Arr[5];
  TemplUses<CEVar, int, int[1]>({}, {}, &i);
}
#endif // PCH_HELPER
