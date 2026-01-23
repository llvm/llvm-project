// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -fopenacc -emit-pch -o %t %s
// RUN: %clang_cc1 %s -fopenacc -include-pch %t -ast-dump-all | FileCheck %s

#ifndef PCH_HELPER
#define PCH_HELPER

void NormalFunc(int i, float f) {
  // CHECK: FunctionDecl{{.*}}NormalFunc
  // CHECK-NEXT: ParmVarDecl
  // CHECK-NEXT: ParmVarDecl
  // CHECK-NEXT: CompoundStmt

#pragma acc parallel loop reduction(+: i)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} parallel loop
  // CHECK-NEXT: reduction clause Operator: +
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int' lvalue ParmVar{{.*}} 'i' 'int'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc serial loop reduction(*: f)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} serial loop
  // CHECK-NEXT: reduction clause Operator: *
  // CHECK-NEXT: DeclRefExpr{{.*}} 'float' lvalue ParmVar{{.*}} 'f' 'float'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt


#pragma acc kernels loop reduction(max: i)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} kernels loop
  // CHECK-NEXT: reduction clause Operator: max
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int' lvalue ParmVar{{.*}} 'i' 'int'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

  }

template<typename T>
void TemplFunc() {
  // CHECK: FunctionTemplateDecl{{.*}}TemplFunc
  // CHECK-NEXT: TemplateTypeParmDecl

  // Match the prototype:
  // CHECK-NEXT: FunctionDecl{{.*}}TemplFunc
  // CHECK-NEXT: CompoundStmt

  T t;
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} t 'T'

#pragma acc parallel loop reduction(+: t)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} parallel loop
  // CHECK-NEXT: reduction clause Operator: +
  // CHECK-NEXT: DeclRefExpr{{.*}} 'T' lvalue Var{{.*}} 't' 'T'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc serial loop reduction(*: T::SomeFloat)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} serial loop
  // CHECK-NEXT: reduction clause Operator: *
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'T'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

  typename T::IntTy i;
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} i 'typename T::IntTy'

#pragma acc kernels loop reduction(max: i)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} kernels loop
  // CHECK-NEXT: reduction clause Operator: max
  // CHECK-NEXT: DeclRefExpr{{.*}} 'typename T::IntTy' lvalue Var{{.*}} 'i' 'typename T::IntTy'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

  // Match the instantiation:

  // CHECK: FunctionDecl{{.*}}TemplFunc 'void ()' implicit_instantiation
  // CHECK-NEXT: TemplateArgument type 'InstTy'
  // CHECK-NEXT: RecordType{{.*}} 'InstTy'
  // CHECK-NEXT: CXXRecord{{.*}} 'InstTy'
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} used t 'InstTy'
  // CHECK-NEXT: CXXConstructExpr{{.*}} 'InstTy' 'void () noexcept'

  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} parallel loop
  // CHECK-NEXT: reduction clause Operator: +
  // CHECK-NEXT: DeclRefExpr{{.*}} 'InstTy' lvalue Var{{.*}} 't' 'InstTy'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} serial loop
  // CHECK-NEXT: reduction clause Operator: *
  // CHECK-NEXT: DeclRefExpr{{.*}} 'const float' lvalue Var{{.*}} 'SomeFloat' 'const float'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'InstTy'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} i 'typename InstTy::IntTy':'int'

  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} kernels loop
  // CHECK-NEXT: reduction clause Operator: max
  // CHECK-NEXT: DeclRefExpr{{.*}} 'typename InstTy::IntTy':'int' lvalue Var{{.*}} 'i' 'typename InstTy::IntTy':'int'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

}

struct InstTy {
  using IntTy = int;
  static constexpr float SomeFloat = 5.0;
};

void Instantiate() {
  TemplFunc<InstTy>();
}
#endif // PCH_HELPER
