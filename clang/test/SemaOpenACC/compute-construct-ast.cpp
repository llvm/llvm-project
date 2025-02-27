// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -fopenacc -emit-pch -o %t %s
// RUN: %clang_cc1 %s -fopenacc -include-pch %t -ast-dump-all | FileCheck %s

#ifndef PCH_HELPER
#define PCH_HELPER

void NormalFunc() {
  // CHECK-LABEL: NormalFunc
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: OpenACCComputeConstruct {{.*}}parallel
  // CHECK-NEXT: default(none)
  // CHECK-NEXT: CompoundStmt
#pragma acc parallel default(none)
  {
#pragma acc parallel
  // CHECK-NEXT: OpenACCComputeConstruct {{.*}}parallel
  // CHECK-NEXT: OpenACCComputeConstruct {{.*}}parallel
  // CHECK-NEXT: default(present)
  // CHECK-NEXT: CompoundStmt
#pragma acc parallel default(present)
    {}
  }
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
#pragma acc parallel default(none)
  {
    typename T::type I;
  }

#pragma acc serial default(present)
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
  // CHECK-NEXT: default(none)
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} I 'typename T::type'
  // CHECK-NEXT: OpenACCComputeConstruct {{.*}}serial
  // CHECK-NEXT: default(present)
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
  // CHECK-NEXT: default(none)
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} I 'typename S::type':'int'
  // CHECK-NEXT: OpenACCComputeConstruct {{.*}}serial
  // CHECK-NEXT: default(present)
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

struct HasCtor { HasCtor(); operator int(); ~HasCtor();};

void useCtorType() {
  // CHECK-LABEL: useCtorType
  // CHECK-NEXT: CompoundStmt

#pragma acc kernels num_workers(HasCtor{})
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} kernels
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'int'
  // CHECK-NEXT: MemberExpr{{.*}}.operator int
  // CHECK-NEXT: MaterializeTemporaryExpr{{.*}}'HasCtor'
  // CHECK-NEXT: CXXBindTemporaryExpr{{.*}}'HasCtor'
  // CHECK-NEXT: CXXTemporaryObjectExpr{{.*}}'HasCtor'

  while(true);
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt
}
#endif
