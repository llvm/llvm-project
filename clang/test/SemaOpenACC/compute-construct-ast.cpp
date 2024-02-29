// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

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
}

template<typename T>
void TemplFunc() {
#pragma acc parallel
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
}

struct S {
  using type = int;
};

void use() {
  TemplFunc<S>();
}
