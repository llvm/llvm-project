
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

#pragma acc exit data delete(GlobalArray, PointerParam[Global])
  // CHECK-NEXT: OpenACCExitDataConstruct{{.*}} exit data
  // CHECK-NEXT: delete clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'short[5]' lvalue Var{{.*}}'GlobalArray' 'short[5]'
  // CHECK-NEXT: ArraySubscriptExpr{{.*}}'float' lvalue
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'float *' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}}'float *' lvalue ParmVar{{.*}}'PointerParam' 'float *'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue Var{{.*}}'Global' 'int'
}

template<typename T>
void TemplUses(T t) {
  // CHECK-NEXT: FunctionTemplateDecl
  // CHECK-NEXT: TemplateTypeParmDecl{{.*}}typename depth 0 index 0 T
  // CHECK-NEXT: FunctionDecl{{.*}} TemplUses 'void (T)'
  // CHECK-NEXT: ParmVarDecl{{.*}} referenced t 'T'
  // CHECK-NEXT: CompoundStmt

#pragma acc exit data delete(t)
  // CHECK-NEXT: OpenACCExitDataConstruct{{.*}} exit data
  // CHECK-NEXT: delete clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'T' lvalue ParmVar{{.*}} 't' 'T'

  // Check the instantiated versions of the above.
  // CHECK-NEXT: FunctionDecl{{.*}} used TemplUses 'void (int)' implicit_instantiation
  // CHECK-NEXT: TemplateArgument type 'int'
  // CHECK-NEXT: BuiltinType{{.*}} 'int'
  // CHECK-NEXT: ParmVarDecl{{.*}} used t 'int'
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCExitDataConstruct{{.*}} exit data
  // CHECK-NEXT: delete clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue ParmVar{{.*}} 't' 'int'
}

void Inst() {
  int i;
  TemplUses(i);
}
#endif
