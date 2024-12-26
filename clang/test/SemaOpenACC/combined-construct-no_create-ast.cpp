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

#pragma acc parallel loop no_create(GlobalArray, PointerParam[Global])
  for (unsigned i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} parallel loop
  // CHECK-NEXT: no_create clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'short[5]' lvalue Var{{.*}}'GlobalArray' 'short[5]'
  // CHECK-NEXT: ArraySubscriptExpr{{.*}}'float' lvalue
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'float *' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}}'float *' lvalue ParmVar{{.*}}'PointerParam' 'float *'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue Var{{.*}}'Global' 'int'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt
}

template<auto &NTTP, typename T, typename U>
void TemplUses(T t, U u) {
  // CHECK-NEXT: FunctionTemplateDecl
  // CHECK-NEXT: NonTypeTemplateParmDecl {{.*}}referenced 'auto &' depth 0 index 0 NTTP
  // CHECK-NEXT: TemplateTypeParmDecl{{.*}}typename depth 0 index 1 T
  // CHECK-NEXT: TemplateTypeParmDecl{{.*}}typename depth 0 index 2 U
  // CHECK-NEXT: FunctionDecl{{.*}} TemplUses 'void (T, U)'
  // CHECK-NEXT: ParmVarDecl{{.*}} referenced t 'T'
  // CHECK-NEXT: ParmVarDecl{{.*}} referenced u 'U'
  // CHECK-NEXT: CompoundStmt


#pragma acc parallel loop no_create(t) present(NTTP, u)
  for (unsigned i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} parallel loop
  // CHECK-NEXT: no_create clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'T' lvalue ParmVar{{.*}} 't' 'T'
  // CHECK-NEXT: present clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'auto' lvalue NonTypeTemplateParm{{.*}} 'NTTP' 'auto &'
  // CHECK-NEXT: DeclRefExpr{{.*}}'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

  // Check the instantiated versions of the above.
  // CHECK-NEXT: FunctionDecl{{.*}} used TemplUses 'void (int, int *)' implicit_instantiation
  // CHECK-NEXT: TemplateArgument decl
  // CHECK-NEXT: Var{{.*}} 'CEVar' 'const unsigned int'
  // CHECK-NEXT: TemplateArgument type 'int'
  // CHECK-NEXT: BuiltinType{{.*}} 'int'
  // CHECK-NEXT: TemplateArgument type 'int *'
  // CHECK-NEXT: PointerType{{.*}} 'int *'
  // CHECK-NEXT: BuiltinType{{.*}} 'int'
  // CHECK-NEXT: ParmVarDecl{{.*}} used t 'int'
  // CHECK-NEXT: ParmVarDecl{{.*}} used u 'int *'
  // CHECK-NEXT: CompoundStmt

// #pragma acc parallel loop no_create(t) present(NTTP, u)
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} parallel loop
  // CHECK-NEXT: no_create clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue ParmVar{{.*}} 't' 'int'
  // CHECK-NEXT: present clause
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
  TemplUses<CEVar>(i, &i);
}
#endif
