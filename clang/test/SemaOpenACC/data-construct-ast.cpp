// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -fopenacc -emit-pch -o %t %s
// RUN: %clang_cc1 %s -fopenacc -include-pch %t -ast-dump-all | FileCheck %s

#ifndef PCH_HELPER
#define PCH_HELPER

void NormalFunc() {
  // CHECK-LABEL: NormalFunc
  // CHECK-NEXT: CompoundStmt

  int Var;
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl

  // TODO OpenACC: these constructs require the clauses to be legal, but we
  // don't have the clauses implemented yet.  As we implement them, they needed
  // to be added to the 'check' lines.

#pragma acc data default(none)
  while (Var);
  // CHECK-NEXT: OpenACCDataConstruct{{.*}}data
  // CHECK-NEXT: default(none)
  // CHECK-NEXT: WhileStmt
  // CHECK: NullStmt
#pragma acc enter data copyin(Var)
  // CHECK-NEXT: OpenACCEnterDataConstruct{{.*}} enter data
  // CHECK-NEXT: copyin clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'Var' 'int'
#pragma acc exit data copyout(Var)
  // CHECK-NEXT: OpenACCExitDataConstruct{{.*}} exit data
  // CHECK-NEXT: copyout clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'Var' 'int'
#pragma acc host_data use_device(Var)
  while (Var);
  // CHECK-NEXT: OpenACCHostDataConstruct{{.*}} host_data
  // CHECK-NEXT: use_device clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'Var' 'int'
  // CHECK-NEXT: WhileStmt
  // CHECK: NullStmt
}

template<typename T>
void TemplFunc() {
  // CHECK-LABEL: FunctionTemplateDecl {{.*}}TemplFunc
  // CHECK-NEXT: TemplateTypeParmDecl
  // CHECK-NEXT: FunctionDecl{{.*}}TemplFunc
  // CHECK-NEXT: CompoundStmt

  T Var;
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl

#pragma acc data default(none)
  while (Var);
  // CHECK-NEXT: OpenACCDataConstruct{{.*}}data
  // CHECK-NEXT: default(none)
  // CHECK-NEXT: WhileStmt
  // CHECK: NullStmt
#pragma acc enter data copyin(Var)
  // CHECK-NEXT: OpenACCEnterDataConstruct{{.*}} enter data
  // CHECK-NEXT: copyin clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'Var' 'T'
#pragma acc exit data copyout(Var)
  // CHECK-NEXT: OpenACCExitDataConstruct{{.*}} exit data
  // CHECK-NEXT: copyout clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'Var' 'T'
#pragma acc host_data use_device(Var)
  while (Var);
  // CHECK-NEXT: OpenACCHostDataConstruct{{.*}} host_data
  // CHECK-NEXT: use_device clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'Var' 'T'
  // CHECK-NEXT: WhileStmt
  // CHECK: NullStmt

  // Instantiation:
  // CHECK-NEXT: FunctionDecl{{.*}} TemplFunc 'void ()' implicit_instantiation
  // CHECK-NEXT: TemplateArgument type 'int'
  // CHECK-NEXT: BuiltinType{{.*}} 'int'
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl

  // CHECK-NEXT: OpenACCDataConstruct{{.*}}data
  // CHECK-NEXT: default(none)
  // CHECK-NEXT: WhileStmt
  // CHECK: NullStmt

  // CHECK-NEXT: OpenACCEnterDataConstruct{{.*}} enter data
  // CHECK-NEXT: copyin clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'Var' 'int'

  // CHECK-NEXT: OpenACCExitDataConstruct{{.*}} exit data
  // CHECK-NEXT: copyout clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'Var' 'int'

  // CHECK-NEXT: OpenACCHostDataConstruct{{.*}} host_data
  // CHECK-NEXT: use_device clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'Var' 'int'
  // CHECK-NEXT: WhileStmt
  // CHECK: NullStmt
}
void use() {
  TemplFunc<int>();
}
#endif
