// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -fopenacc -emit-pch -o %t %s
// RUN: %clang_cc1 %s -fopenacc -include-pch %t -ast-dump-all | FileCheck %s
#ifndef PCH_HELPER
#define PCH_HELPER

void Uses() {
  // CHECK: FunctionDecl{{.*}}Uses
  // CHECK-NEXT: CompoundStmt

  int I;
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl

#pragma acc exit data copyout(I) finalize
  // CHECK-NEXT: OpenACCExitDataConstruct{{.*}}exit data
  // CHECK-NEXT: copyout clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'I' 'int'
  // CHECK-NEXT: finalize clause
}

template<typename T>
void TemplUses() {
  // CHECK: FunctionTemplateDecl{{.*}}TemplUses
  // CHECK-NEXT: TemplateTypeParmDecl{{.*}}T
  // CHECK-NEXT: FunctionDecl{{.*}}TemplUses
  // CHECK-NEXT: CompoundStmt

  T I;
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl

#pragma acc exit data copyout(I) finalize
  // CHECK-NEXT: OpenACCExitDataConstruct{{.*}}exit data
  // CHECK-NEXT: copyout clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'I' 'T'
  // CHECK-NEXT: finalize clause

  // Instantiations
  // CHECK-NEXT: FunctionDecl{{.*}} TemplUses 'void ()' implicit_instantiation
  // CHECK-NEXT: TemplateArgument type 'int'
  // CHECK-NEXT: BuiltinType{{.*}} 'int'
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl

  // CHECK-NEXT: OpenACCExitDataConstruct{{.*}}exit data
  // CHECK-NEXT: copyout clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'I' 'int'
  // CHECK-NEXT: finalize clause
}
void Inst() {
  TemplUses<int>();
}


#endif // PCH_HELPER
