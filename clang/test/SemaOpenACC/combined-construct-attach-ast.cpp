// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -fopenacc -emit-pch -o %t %s
// RUN: %clang_cc1 %s -fopenacc -include-pch %t -ast-dump-all | FileCheck %s

#ifndef PCH_HELPER
#define PCH_HELPER

void NormalUses(float *PointerParam) {
  // CHECK: FunctionDecl{{.*}}NormalUses
  // CHECK: ParmVarDecl
  // CHECK-NEXT: CompoundStmt

#pragma acc parallel loop attach(PointerParam) deviceptr(PointerParam)
  for (unsigned i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} parallel loop
  // CHECK-NEXT: attach clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'float *' lvalue ParmVar{{.*}} 'PointerParam' 'float *'
  // CHECK-NEXT: deviceptr clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'float *' lvalue ParmVar{{.*}} 'PointerParam' 'float *'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt
}

template<typename T>
void TemplUses(T *PointerParam) {
  // CHECK-NEXT: FunctionTemplateDecl
  // CHECK-NEXT: TemplateTypeParmDecl{{.*}}typename depth 0 index 0 T
  // CHECK-NEXT: FunctionDecl{{.*}} TemplUses 'void (T *)'
  // CHECK-NEXT: ParmVarDecl{{.*}} referenced PointerParam 'T *'
  // CHECK-NEXT: CompoundStmt

#pragma acc parallel loop attach(PointerParam) deviceptr(PointerParam)
  for (unsigned i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} parallel loop
  // CHECK-NEXT: attach clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'T *' lvalue ParmVar{{.*}} 'PointerParam' 'T *'
  // CHECK-NEXT: deviceptr clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'T *' lvalue ParmVar{{.*}} 'PointerParam' 'T *'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

  // Check the instantiated versions of the above.
  // CHECK-NEXT: FunctionDecl{{.*}} used TemplUses 'void (int *)' implicit_instantiation
  // CHECK-NEXT: TemplateArgument type 'int'
  // CHECK-NEXT: BuiltinType{{.*}} 'int'
  // CHECK-NEXT: ParmVarDecl{{.*}} used PointerParam 'int *'
  // CHECK-NEXT: CompoundStmt

//#pragma acc parallel loop attach(PointerParam) deviceptr(PointerParam)
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} parallel loop
  // CHECK-NEXT: attach clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'int *' lvalue ParmVar{{.*}} 'PointerParam' 'int *'
  // CHECK-NEXT: deviceptr clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'int *' lvalue ParmVar{{.*}} 'PointerParam' 'int *'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

}

void Inst() {
  int i;
  TemplUses(&i);
}
#endif
