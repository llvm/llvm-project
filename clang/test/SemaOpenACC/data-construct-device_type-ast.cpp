// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -fopenacc -emit-pch -o %t %s
// RUN: %clang_cc1 %s -fopenacc -include-pch %t -ast-dump-all | FileCheck %s
#ifndef PCH_HELPER
#define PCH_HELPER

template<typename T>
void TemplUses() {
  // CHECK: FunctionTemplateDecl{{.*}}TemplUses
  // CHECK-NEXT: TemplateTypeParmDecl{{.*}}T
  // CHECK-NEXT: FunctionDecl{{.*}}TemplUses
  // CHECK-NEXT: CompoundStmt

#pragma acc data default(none) device_type(T) dtype(T)
  ;
  // CHECK-NEXT: OpenACCDataConstruct{{.*}} data
  // CHECK-NEXT: default(none)
  // CHECK-NEXT: device_type(T)
  // CHECK-NEXT: dtype(T)
  // CHECK-NEXT: NullStmt

  // Instantiations
  // CHECK-NEXT: FunctionDecl{{.*}} TemplUses 'void ()' implicit_instantiation
  // CHECK-NEXT: TemplateArgument type 'int'
  // CHECK-NEXT: BuiltinType{{.*}} 'int'
  // CHECK-NEXT: CompoundStmt

  // Argument to 'device-type' is just an identifier, so we don't transform it.
  // CHECK-NEXT: OpenACCDataConstruct{{.*}} data
  // CHECK-NEXT: default(none)
  // CHECK-NEXT: device_type(T)
  // CHECK-NEXT: dtype(T)
  // CHECK-NEXT: NullStmt
}
void Inst() {
  TemplUses<int>();
}

#endif // PCH_HELPER
