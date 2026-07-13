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

#pragma acc host_data use_device(I) if_present
  ;
  // CHECK-NEXT: OpenACCHostDataConstruct{{.*}}host_data
  // CHECK-NEXT: use_device clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'I' 'int'
  // CHECK-NEXT: if_present clause
  // CHECK-NEXT: NullStmt
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

#pragma acc host_data use_device(I) if_present
  ;
  // CHECK-NEXT: OpenACCHostDataConstruct{{.*}}host_data
  // CHECK-NEXT: use_device clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'I' 'T'
  // CHECK-NEXT: if_present clause
  // CHECK-NEXT: NullStmt

  // Instantiations
  // CHECK-NEXT: FunctionDecl{{.*}} TemplUses 'void ()' implicit_instantiation
  // CHECK-NEXT: TemplateArgument type 'int'
  // CHECK-NEXT: BuiltinType{{.*}} 'int'
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl

  // CHECK-NEXT: OpenACCHostDataConstruct{{.*}}host_data
  // CHECK-NEXT: use_device clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'I' 'int'
  // CHECK-NEXT: if_present clause
  // CHECK-NEXT: NullStmt
}
void Inst() {
  TemplUses<int>();
}


#endif // PCH_HELPER
