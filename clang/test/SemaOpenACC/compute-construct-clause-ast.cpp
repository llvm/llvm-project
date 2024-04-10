// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -fopenacc -emit-pch -o %t %s
// RUN: %clang_cc1 %s -fopenacc -include-pch %t -ast-dump-all | FileCheck %s

#ifndef PCH_HELPER
#define PCH_HELPER
void NormalFunc() {
  // CHECK: FunctionDecl{{.*}}NormalFunc
  // CHECK-NEXT: CompoundStmt
#pragma acc parallel default(none)
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: default(none)
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc serial default(present)
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}serial
  // CHECK-NEXT: default(present)
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt
}

template<typename T>
void TemplFunc() {
  // CHECK: FunctionTemplateDecl{{.*}}TemplFunc
  // CHECK-NEXT: TemplateTypeParmDecl

  // Match the prototype:
  // CHECK-NEXT: FunctionDecl{{.*}}TemplFunc
  // CHECK-NEXT: CompoundStmt

#pragma acc kernels default(none)
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}kernels
  // CHECK-NEXT: default(none)
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel default(present)
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: default(present)
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

  // Match the instantiation:
  // CHECK: FunctionDecl{{.*}}TemplFunc{{.*}}implicit_instantiation
  // CHECK-NEXT: TemplateArgument type 'int'
  // CHECK-NEXT: BuiltinType
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}kernels
  // CHECK-NEXT: default(none)
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: default(present)
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt
}

void Instantiate() {
  TemplFunc<int>();
}
#endif
