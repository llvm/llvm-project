// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -fopenacc -emit-pch -o %t %s
// RUN: %clang_cc1 %s -fopenacc -include-pch %t -ast-dump-all | FileCheck %s

#ifndef PCH_HELPER
#define PCH_HELPER

void NormalFunc() {
  // FIXME: Add a test once we have clauses for this.
  // CHECK-LABEL: NormalFunc
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: OpenACCComputeConstruct {{.*}}parallel
  // CHECK-NEXT: CompoundStmt
#pragma acc parallel
  {
#pragma acc parallel
  // CHECK-NEXT: OpenACCComputeConstruct {{.*}}parallel
  // CHECK-NEXT: OpenACCComputeConstruct {{.*}}parallel
  // CHECK-NEXT: CompoundStmt
#pragma acc parallel
    {}
  }
  // FIXME: Add a test once we have clauses for this.
  // CHECK-NEXT: OpenACCComputeConstruct {{.*}}serial
  // CHECK-NEXT: CompoundStmt
#pragma acc serial
  {
#pragma acc serial
  // CHECK-NEXT: OpenACCComputeConstruct {{.*}}serial
  // CHECK-NEXT: OpenACCComputeConstruct {{.*}}serial
  // CHECK-NEXT: CompoundStmt
#pragma acc serial
    {}
  }
  // FIXME: Add a test once we have clauses for this.
  // CHECK-NEXT: OpenACCComputeConstruct {{.*}}kernels
  // CHECK-NEXT: CompoundStmt
#pragma acc kernels
  {
#pragma acc kernels
  // CHECK-NEXT: OpenACCComputeConstruct {{.*}}kernels
  // CHECK-NEXT: OpenACCComputeConstruct {{.*}}kernels
  // CHECK-NEXT: CompoundStmt
#pragma acc kernels
    {}
  }
}

template<typename T>
void TemplFunc() {
#pragma acc parallel
  {
    typename T::type I;
  }

#pragma acc serial
  {
    typename T::type I;
  }

#pragma acc kernels
  {
    typename T::type I;
  }

  // CHECK-LABEL: FunctionTemplateDecl {{.*}}TemplFunc
  // CHECK-NEXT: TemplateTypeParmDecl

  // Template Pattern:
  // CHECK-NEXT: FunctionDecl
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: OpenACCComputeConstruct {{.*}}parallel
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} I 'typename T::type'
  // CHECK-NEXT: OpenACCComputeConstruct {{.*}}serial
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} I 'typename T::type'
  // CHECK-NEXT: OpenACCComputeConstruct {{.*}}kernels
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} I 'typename T::type'

  // Check instantiation.
  // CHECK-LABEL: FunctionDecl{{.*}} used TemplFunc 'void ()' implicit_instantiation
  // CHECK-NEXT: TemplateArgument type 'S'
  // CHECK-NEXT: RecordType
  // CHECK-NEXT: CXXRecord
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: OpenACCComputeConstruct {{.*}}parallel
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} I 'typename S::type':'int'
  // CHECK-NEXT: OpenACCComputeConstruct {{.*}}serial
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} I 'typename S::type':'int'
  // CHECK-NEXT: OpenACCComputeConstruct {{.*}}kernels
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} I 'typename S::type':'int'
}

struct S {
  using type = int;
};

void use() {
  TemplFunc<S>();
}
#endif

