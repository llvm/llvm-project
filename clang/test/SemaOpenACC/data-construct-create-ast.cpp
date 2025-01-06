// RUN: %clang_cc1 %s -fopenacc -Wno-openacc-deprecated-clause-alias -ast-dump | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -fopenacc -Wno-openacc-deprecated-clause-alias -emit-pch -o %t %s
// RUN: %clang_cc1 %s -fopenacc -Wno-openacc-deprecated-clause-alias -include-pch %t -ast-dump-all | FileCheck %s

#ifndef PCH_HELPER
#define PCH_HELPER

int Global;
short GlobalArray[5];
void NormalUses(float *PointerParam) {
  // CHECK: FunctionDecl{{.*}}NormalUses
  // CHECK: ParmVarDecl
  // CHECK-NEXT: CompoundStmt

#pragma acc data create(GlobalArray) pcreate(zero:PointerParam[Global]) present_or_create(Global)
  ;
  // CHECK-NEXT: OpenACCDataConstruct{{.*}} data
  // CHECK-NEXT: create clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'short[5]' lvalue Var{{.*}}'GlobalArray' 'short[5]'
  // CHECK-NEXT: pcreate clause : zero
  // CHECK-NEXT: ArraySubscriptExpr{{.*}}'float' lvalue
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'float *' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}}'float *' lvalue ParmVar{{.*}}'PointerParam' 'float *'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue Var{{.*}}'Global' 'int'
  // CHECK-NEXT: present_or_create clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue Var{{.*}}'Global' 'int'
  // CHECK-NEXT: NullStmt

#pragma acc enter data create(GlobalArray) pcreate(zero:PointerParam[Global]) present_or_create(Global)
  // CHECK-NEXT: OpenACCEnterDataConstruct{{.*}} enter data
  // CHECK-NEXT: create clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'short[5]' lvalue Var{{.*}}'GlobalArray' 'short[5]'
  // CHECK-NEXT: pcreate clause : zero
  // CHECK-NEXT: ArraySubscriptExpr{{.*}}'float' lvalue
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'float *' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}}'float *' lvalue ParmVar{{.*}}'PointerParam' 'float *'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue Var{{.*}}'Global' 'int'
  // CHECK-NEXT: present_or_create clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue Var{{.*}}'Global' 'int'
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

#pragma acc data create(t) pcreate(zero: NTTP, u) present_or_create(u[0:t])
  ;
  // CHECK-NEXT: OpenACCDataConstruct{{.*}} data
  // CHECK-NEXT: create clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'T' lvalue ParmVar{{.*}} 't' 'T'
  // CHECK-NEXT: pcreate clause : zero
  // CHECK-NEXT: DeclRefExpr{{.*}}'auto' lvalue NonTypeTemplateParm{{.*}} 'NTTP' 'auto &'
  // CHECK-NEXT: DeclRefExpr{{.*}}'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: present_or_create clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 0
  // CHECK-NEXT: DeclRefExpr{{.*}}'T' lvalue ParmVar{{.*}} 't' 'T'
  // CHECK-NEXT: NullStmt

#pragma acc enter data create(t) pcreate(zero: NTTP, u) present_or_create(u[0:t])
  // CHECK-NEXT: OpenACCEnterDataConstruct{{.*}} enter data
  // CHECK-NEXT: create clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'T' lvalue ParmVar{{.*}} 't' 'T'
  // CHECK-NEXT: pcreate clause : zero
  // CHECK-NEXT: DeclRefExpr{{.*}}'auto' lvalue NonTypeTemplateParm{{.*}} 'NTTP' 'auto &'
  // CHECK-NEXT: DeclRefExpr{{.*}}'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: present_or_create clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 0
  // CHECK-NEXT: DeclRefExpr{{.*}}'T' lvalue ParmVar{{.*}} 't' 'T'

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

  // CHECK-NEXT: OpenACCDataConstruct{{.*}} data
  // CHECK-NEXT: create clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue ParmVar{{.*}} 't' 'int'
  // CHECK-NEXT: pcreate clause : zero
  // CHECK-NEXT: SubstNonTypeTemplateParmExpr{{.*}}'const unsigned int' lvalue
  // CHECK-NEXT: NonTypeTemplateParmDecl{{.*}} referenced 'auto &' depth 0 index 0 NTTP
  // CHECK-NEXT: DeclRefExpr{{.*}}'const unsigned int' lvalue Var{{.*}} 'CEVar' 'const unsigned int'
  // CHECK-NEXT: DeclRefExpr{{.*}}'int *' lvalue ParmVar{{.*}} 'u' 'int *'
  // CHECK-NEXT: present_or_create clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int *' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int *' lvalue ParmVar{{.*}} 'u' 'int *'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 0
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue ParmVar{{.*}} 't' 'int'
  // CHECK-NEXT: NullStmt

  // CHECK-NEXT: OpenACCEnterDataConstruct{{.*}} enter data
  // CHECK-NEXT: create clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue ParmVar{{.*}} 't' 'int'
  // CHECK-NEXT: pcreate clause : zero
  // CHECK-NEXT: SubstNonTypeTemplateParmExpr{{.*}}'const unsigned int' lvalue
  // CHECK-NEXT: NonTypeTemplateParmDecl{{.*}} referenced 'auto &' depth 0 index 0 NTTP
  // CHECK-NEXT: DeclRefExpr{{.*}}'const unsigned int' lvalue Var{{.*}} 'CEVar' 'const unsigned int'
  // CHECK-NEXT: DeclRefExpr{{.*}}'int *' lvalue ParmVar{{.*}} 'u' 'int *'
  // CHECK-NEXT: present_or_create clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int *' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int *' lvalue ParmVar{{.*}} 'u' 'int *'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 0
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue ParmVar{{.*}} 't' 'int'
}

void Inst() {
  static constexpr unsigned CEVar = 1;
  int i;
  TemplUses<CEVar>(i, &i);
}
#endif
