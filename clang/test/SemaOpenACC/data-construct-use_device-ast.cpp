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

#pragma acc host_data use_device(GlobalArray, PointerParam)
  ;
  // CHECK-NEXT: OpenACCHostDataConstruct{{.*}} host_data
  // CHECK-NEXT: use_device clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'short[5]' lvalue Var{{.*}}'GlobalArray' 'short[5]'
  // CHECK-NEXT: DeclRefExpr{{.*}}'float *' lvalue ParmVar{{.*}}'PointerParam' 'float *'
  // CHECK-NEXT: NullStmt
}

template<typename T>
void TemplUses(T t) {
  // CHECK-NEXT: FunctionTemplateDecl
  // CHECK-NEXT: TemplateTypeParmDecl{{.*}}typename depth 0 index 0 T
  // CHECK-NEXT: FunctionDecl{{.*}} TemplUses 'void (T)'
  // CHECK-NEXT: ParmVarDecl{{.*}} referenced t 'T'
  // CHECK-NEXT: CompoundStmt

#pragma acc host_data use_device(t)
  ;
  // CHECK-NEXT: OpenACCHostDataConstruct{{.*}} host_data
  // CHECK-NEXT: use_device clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'T' lvalue ParmVar{{.*}} 't' 'T'
  // CHECK-NEXT: NullStmt

  // Check the instantiated versions of the above.
  // CHECK-NEXT: FunctionDecl{{.*}} used TemplUses 'void (int)' implicit_instantiation
  // CHECK-NEXT: TemplateArgument type 'int'
  // CHECK-NEXT: BuiltinType{{.*}} 'int'
  // CHECK-NEXT: ParmVarDecl{{.*}} used t 'int'
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCHostDataConstruct{{.*}} host_data
  // CHECK-NEXT: use_device clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue ParmVar{{.*}} 't' 'int'
  // CHECK-NEXT: NullStmt
}

void Inst() {
  int i;
  TemplUses(i);
}
#endif
