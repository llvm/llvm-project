// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -fopenacc -emit-pch -o %t %s
// RUN: %clang_cc1 %s -fopenacc -include-pch %t -ast-dump-all | FileCheck %s
#ifndef PCH_HELPER
#define PCH_HELPER

struct SomeS{};
void NormalUses() {
  // CHECK: FunctionDecl{{.*}}NormalUses
  // CHECK-NEXT: CompoundStmt

  SomeS SomeImpl;
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} SomeImpl 'SomeS'
  // CHECK-NEXT: CXXConstructExpr
  bool SomeVar;
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} SomeVar 'bool'

#pragma acc parallel device_type(SomeS) dtype(SomeImpl)
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: device_type(SomeS)
  // CHECK-NEXT: dtype(SomeImpl)
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt
#pragma acc parallel device_type(SomeVar) dtype(int)
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: device_type(SomeVar)
  // CHECK-NEXT: dtype(int)
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt
#pragma acc parallel device_type(private) dtype(struct)
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: device_type(private)
  // CHECK-NEXT: dtype(struct)
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt
#pragma acc parallel device_type(private) dtype(class)
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: device_type(private)
  // CHECK-NEXT: dtype(class)
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt
#pragma acc parallel device_type(float) dtype(*)
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: device_type(float)
  // CHECK-NEXT: dtype(*)
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt
#pragma acc parallel device_type(float, int) dtype(*)
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: device_type(float, int)
  // CHECK-NEXT: dtype(*)
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt
}

template<typename T>
void TemplUses() {
  // CHECK-NEXT: FunctionTemplateDecl{{.*}}TemplUses
  // CHECK-NEXT: TemplateTypeParmDecl{{.*}}T
  // CHECK-NEXT: FunctionDecl{{.*}}TemplUses
  // CHECK-NEXT: CompoundStmt
#pragma acc parallel device_type(T) dtype(T)
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: device_type(T)
  // CHECK-NEXT: dtype(T)
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt


  // Instantiations
  // CHECK-NEXT: FunctionDecl{{.*}} TemplUses 'void ()' implicit_instantiation
  // CHECK-NEXT: TemplateArgument type 'int'
  // CHECK-NEXT: BuiltinType{{.*}} 'int'
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: device_type(T)
  // CHECK-NEXT: dtype(T)
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt
}

void Inst() {
  TemplUses<int>();
}
#endif // PCH_HELPER
