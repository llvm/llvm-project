// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -fopenacc -emit-pch -o %t %s
// RUN: %clang_cc1 %s -fopenacc -include-pch %t -ast-dump-all | FileCheck %s
#ifndef PCH_HELPER
#define PCH_HELPER

int some_int();

template<typename T>
void TemplUses() {
  // CHECK: FunctionTemplateDecl{{.*}}TemplUses
  // CHECK-NEXT: TemplateTypeParmDecl{{.*}}T
  // CHECK-NEXT: FunctionDecl{{.*}}TemplUses
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl
  T t;

#pragma acc data default(none) async(some_int())
  ;
  // CHECK-NEXT: OpenACCDataConstruct{{.*}}data
  // CHECK-NEXT: default(none)
  // CHECK-NEXT: async clause
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_int' 'int ()'
  // CHECK-NEXT: NullStmt
#pragma acc enter data copyin(t) async(T{})
  // CHECK-NEXT: OpenACCEnterDataConstruct{{.*}}enter data
  // CHECK-NEXT: copyin clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'t' 'T'
  // CHECK-NEXT: async clause
  // CHECK-NEXT: CXXUnresolvedConstructExpr{{.*}} 'T' 'T' list
  // CHECK-NEXT: InitListExpr{{.*}}'void'
#pragma acc exit data copyout(t) async
  // CHECK-NEXT: OpenACCExitDataConstruct{{.*}}exit data
  // CHECK-NEXT: copyout clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'t' 'T'
  // CHECK-NEXT: async clause

  // Instantiations
  // CHECK-NEXT: FunctionDecl{{.*}} TemplUses 'void ()' implicit_instantiation
  // CHECK-NEXT: TemplateArgument type 'int'
  // CHECK-NEXT: BuiltinType{{.*}} 'int'
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl

  // CHECK-NEXT: OpenACCDataConstruct{{.*}}data
  // CHECK-NEXT: default(none)
  // CHECK-NEXT: async clause
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_int' 'int ()'
  // CHECK-NEXT: NullStmt

  // CHECK-NEXT: OpenACCEnterDataConstruct{{.*}}enter data
  // CHECK-NEXT: copyin clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'t' 'int'
  // CHECK-NEXT: async clause
  // CHECK-NEXT: CXXFunctionalCastExpr
  // CHECK-NEXT: InitListExpr{{.*}}'int'

  // CHECK-NEXT: OpenACCExitDataConstruct{{.*}}exit data
  // CHECK-NEXT: copyout clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'t' 'int'
  // CHECK-NEXT: async clause
}
void Inst() {
  TemplUses<int>();
}


#endif // PCH_HELPER
