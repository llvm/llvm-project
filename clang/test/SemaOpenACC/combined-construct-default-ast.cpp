// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -fopenacc -emit-pch -o %t %s
// RUN: %clang_cc1 %s -fopenacc -include-pch %t -ast-dump-all | FileCheck %s

#ifndef PCH_HELPER
#define PCH_HELPER
void NormalFunc() {
  // CHECK-LABEL: NormalFunc
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: OpenACCCombinedConstruct {{.*}}parallel loop
  // CHECK-NEXT: default(none)
#pragma acc parallel loop default(none)
  for (unsigned I = 0; I < 5; ++I) {
#pragma acc kernels loop
  for (unsigned J = 0; J < 5; ++J) {
  }
  // CHECK: OpenACCCombinedConstruct {{.*}}kernels loop
  // CHECK: OpenACCCombinedConstruct {{.*}}serial loop
  // CHECK-NEXT: default(present)
#pragma acc serial loop default(present)
  for (unsigned J = 0; J < 5; ++J) {
  }
  }
}
template<typename T>
void TemplFunc() {
#pragma acc parallel loop default(none)
  for (unsigned i = 0; i < 5; ++i) {
    typename T::type I;
  }

#pragma acc serial loop default(present)
  for (unsigned i = 0; i < 5; ++i) {
    typename T::type I;
  }

  // CHECK-LABEL: FunctionTemplateDecl {{.*}}TemplFunc
  // CHECK-NEXT: TemplateTypeParmDecl

  // Template Pattern:
  // CHECK-NEXT: FunctionDecl
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: OpenACCCombinedConstruct {{.*}}parallel loop
  // CHECK-NEXT: default(none)
  // CHECK: VarDecl{{.*}} I 'typename T::type'

  // CHECK-NEXT: OpenACCCombinedConstruct {{.*}}serial loop
  // CHECK-NEXT: default(present)
  // CHECK: VarDecl{{.*}} I 'typename T::type'

  // Check instantiation.
  // CHECK-LABEL: FunctionDecl{{.*}} used TemplFunc 'void ()' implicit_instantiation
  // CHECK-NEXT: TemplateArgument type 'S'
  // CHECK-NEXT: RecordType
  // CHECK-NEXT: CXXRecord
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: OpenACCCombinedConstruct {{.*}}parallel loop
  // CHECK-NEXT: default(none)
  // CHECK: VarDecl{{.*}} I 'typename S::type':'int'
  // CHECK-NEXT: OpenACCCombinedConstruct {{.*}}serial loop
  // CHECK-NEXT: default(present)
  // CHECK: VarDecl{{.*}} I 'typename S::type':'int'

}
struct S {
  using type = int;
};

void use() {
  TemplFunc<S>();
}

#endif
