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

#pragma acc parallel device_type(default) dtype(nvidia)
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: device_type(default)
  // CHECK-NEXT: dtype(nvidia)
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt
#pragma acc parallel device_type(radeon) dtype(host)
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: device_type(radeon)
  // CHECK-NEXT: dtype(host)
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt
#pragma acc parallel device_type(multicore) dtype(default)
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: device_type(multicore)
  // CHECK-NEXT: dtype(default)
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt
#pragma acc parallel device_type(nvidia) dtype(acc_device_nvidia)
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: device_type(nvidia)
  // CHECK-NEXT: dtype(acc_device_nvidia)
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt
#pragma acc parallel device_type(radeon) dtype(*)
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: device_type(radeon)
  // CHECK-NEXT: dtype(*)
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt
#pragma acc parallel device_type(host, multicore) dtype(*)
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: device_type(host, multicore)
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
#pragma acc parallel device_type(host) dtype(multicore)
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: device_type(host)
  // CHECK-NEXT: dtype(multicore)
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt


  // Instantiations
  // CHECK-NEXT: FunctionDecl{{.*}} TemplUses 'void ()' implicit_instantiation
  // CHECK-NEXT: TemplateArgument type 'int'
  // CHECK-NEXT: BuiltinType{{.*}} 'int'
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: device_type(host)
  // CHECK-NEXT: dtype(multicore)
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt
}

void Inst() {
  TemplUses<int>();
}
#endif // PCH_HELPER
