// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -fopenacc -emit-pch -o %t %s
// RUN: %clang_cc1 %s -fopenacc -include-pch %t -ast-dump-all | FileCheck %s

#ifndef PCH_HELPER
#define PCH_HELPER
void NormalFunc() {
  // CHECK-LABEL: NormalFunc
  // CHECK-NEXT: CompoundStmt

#pragma acc update
  // CHECK-NEXT: OpenACCUpdateConstruct{{.*}}update
}

template<typename T>
void TemplFunc(T t) {
  // CHECK-LABEL: FunctionTemplateDecl {{.*}}TemplFunc
  // CHECK-NEXT: TemplateTypeParmDecl
  // CHECK-NEXT: FunctionDecl{{.*}}TemplFunc
  // CHECK-NEXT: ParmVarDecl{{.*}} t 'T'
  // CHECK-NEXT: CompoundStmt

#pragma acc update
  // CHECK-NEXT: OpenACCUpdateConstruct{{.*}}update

  // Instantiation:
  // CHECK-NEXT: FunctionDecl{{.*}} TemplFunc 'void (SomeStruct)' implicit_instantiation
  // CHECK-NEXT: TemplateArgument type 'SomeStruct'
  // CHECK-NEXT: RecordType{{.*}} 'SomeStruct'
  // CHECK-NEXT: CXXRecord{{.*}} 'SomeStruct'
  // CHECK-NEXT: ParmVarDecl{{.*}} t 'SomeStruct'
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCUpdateConstruct{{.*}}update
}

struct SomeStruct{
  static constexpr unsigned value = 5;
  operator unsigned();
};
void use() {
  TemplFunc(SomeStruct{});
}
#endif
