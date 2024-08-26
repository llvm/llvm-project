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

#pragma acc loop device_type(SomeS) dtype(SomeImpl)
  for(;;){}
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}
  // CHECK-NEXT: device_type(SomeS)
  // CHECK-NEXT: dtype(SomeImpl)
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: CompoundStmt
#pragma acc loop device_type(SomeVar) dtype(int)
  for(;;){}
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}
  // CHECK-NEXT: device_type(SomeVar)
  // CHECK-NEXT: dtype(int)
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: CompoundStmt
#pragma acc loop device_type(private) dtype(struct)
  for(;;){}
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}
  // CHECK-NEXT: device_type(private)
  // CHECK-NEXT: dtype(struct)
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: CompoundStmt
#pragma acc loop device_type(private) dtype(class)
  for(;;){}
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}
  // CHECK-NEXT: device_type(private)
  // CHECK-NEXT: dtype(class)
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: CompoundStmt
#pragma acc loop device_type(float) dtype(*)
  for(;;){}
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}
  // CHECK-NEXT: device_type(float)
  // CHECK-NEXT: dtype(*)
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: CompoundStmt
#pragma acc loop device_type(float, int) dtype(*)
  for(;;){}
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}
  // CHECK-NEXT: device_type(float, int)
  // CHECK-NEXT: dtype(*)
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: CompoundStmt
}

template<typename T>
void TemplUses() {
  // CHECK-NEXT: FunctionTemplateDecl{{.*}}TemplUses
  // CHECK-NEXT: TemplateTypeParmDecl{{.*}}T
  // CHECK-NEXT: FunctionDecl{{.*}}TemplUses
  // CHECK-NEXT: CompoundStmt
#pragma acc loop device_type(T) dtype(T)
  for(;;){}
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}
  // CHECK-NEXT: device_type(T)
  // CHECK-NEXT: dtype(T)
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: CompoundStmt


  // Instantiations
  // CHECK-NEXT: FunctionDecl{{.*}} TemplUses 'void ()' implicit_instantiation
  // CHECK-NEXT: TemplateArgument type 'int'
  // CHECK-NEXT: BuiltinType{{.*}} 'int'
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}
  // CHECK-NEXT: device_type(T)
  // CHECK-NEXT: dtype(T)
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: CompoundStmt
}

void Inst() {
  TemplUses<int>();
}
#endif // PCH_HELPER
