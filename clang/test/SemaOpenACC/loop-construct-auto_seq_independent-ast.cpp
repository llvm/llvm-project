// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -fopenacc -emit-pch -o %t %s
// RUN: %clang_cc1 %s -fopenacc -include-pch %t -ast-dump-all | FileCheck %s
#ifndef PCH_HELPER
#define PCH_HELPER

void NormalUses() {
  // CHECK: FunctionDecl{{.*}}NormalUses
  // CHECK-NEXT: CompoundStmt

#pragma acc loop auto
  for(;;){}
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}
  // CHECK-NEXT: auto clause
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: CompoundStmt

#pragma acc loop seq
  for(;;){}
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}
  // CHECK-NEXT: seq clause
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: CompoundStmt

#pragma acc loop independent
  for(;;){}
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}
  // CHECK-NEXT: independent clause
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: CompoundStmt
}

template<typename T>
void TemplUses() {
  // CHECK: FunctionTemplateDecl{{.*}}TemplUses
  // CHECK-NEXT: TemplateTypeParmDecl
  // CHECK-NEXT: FunctionDecl{{.*}} TemplUses
  // CHECK-NEXT: CompoundStmt

#pragma acc loop auto
  for(;;){}
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}
  // CHECK-NEXT: auto clause
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: CompoundStmt

#pragma acc loop seq
  for(;;){}
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}
  // CHECK-NEXT: seq clause
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: CompoundStmt

#pragma acc loop independent
  for(;;){}
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}
  // CHECK-NEXT: independent clause
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: CompoundStmt

  // Instantiations.
  // CHECK-NEXT: FunctionDecl{{.*}}TemplUses 'void ()' implicit_instantiation
  // CHECK-NEXT: TemplateArgument
  // CHECK-NEXT: BuiltinType
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}
  // CHECK-NEXT: auto clause
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}
  // CHECK-NEXT: seq clause
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}
  // CHECK-NEXT: independent clause
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
