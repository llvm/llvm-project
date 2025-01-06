// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -fopenacc -emit-pch -o %t %s
// RUN: %clang_cc1 %s -fopenacc -include-pch %t -ast-dump-all | FileCheck %s

#ifndef PCH_HELPER
#define PCH_HELPER
void NormalFunc() {
  // CHECK-LABEL: NormalFunc
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: OpenACCDataConstruct {{.*}}data
  // CHECK-NEXT: default(none)
#pragma acc data  default(none)
  // CHECK: OpenACCDataConstruct {{.*}}data
  // CHECK-NEXT: default(present)
#pragma acc data default(present)
    ;
}
template<typename T>
void TemplFunc() {
#pragma acc data default(none)
  for (unsigned i = 0; i < 5; ++i) {
    typename T::type I;
  }

#pragma acc data default(present)
  for (unsigned i = 0; i < 5; ++i) {
    typename T::type I;
  }

  // CHECK-LABEL: FunctionTemplateDecl {{.*}}TemplFunc
  // CHECK-NEXT: TemplateTypeParmDecl

  // Template Pattern:
  // CHECK-NEXT: FunctionDecl
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: OpenACCDataConstruct {{.*}}data
  // CHECK-NEXT: default(none)
  // CHECK: VarDecl{{.*}} I 'typename T::type'

  // CHECK-NEXT: OpenACCDataConstruct {{.*}}data
  // CHECK-NEXT: default(present)
  // CHECK: VarDecl{{.*}} I 'typename T::type'

  // Check instantiation.
  // CHECK-LABEL: FunctionDecl{{.*}} used TemplFunc 'void ()' implicit_instantiation
  // CHECK-NEXT: TemplateArgument type 'S'
  // CHECK-NEXT: RecordType
  // CHECK-NEXT: CXXRecord
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: OpenACCDataConstruct {{.*}}data
  // CHECK-NEXT: default(none)
  // CHECK: VarDecl{{.*}} I 'typename S::type':'int'
  // CHECK-NEXT: OpenACCDataConstruct {{.*}}data
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
