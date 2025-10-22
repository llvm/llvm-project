// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -fopenacc -emit-pch -o %t %s
// RUN: %clang_cc1 %s -fopenacc -include-pch %t -ast-dump-all | FileCheck %s

#ifndef PCH_HELPER
#define PCH_HELPER

template<typename T>
void TemplFunc() {
  // CHECK: FunctionTemplateDecl{{.*}}TemplFunc
  // CHECK-NEXT: TemplateTypeParmDecl

  // Match the prototype:
  // CHECK-NEXT: FunctionDecl{{.*}}TemplFunc
  // CHECK-NEXT: CompoundStmt

#pragma acc serial loop self
  for (unsigned i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}}serial loop
  // CHECK-NEXT: self clause
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc kernels loop self(T::SomeFloat)
  for (unsigned i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}}kernels loop
  // CHECK-NEXT: self clause
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'T'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc parallel loop self(T::SomeFloat) if (T::SomeFloat)
  for (unsigned i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}}parallel loop
  // CHECK-NEXT: self clause
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'T'
  // CHECK-NEXT: if clause
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'T'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc serial loop if(T::SomeFloat) self(T::SomeFloat)
  for (unsigned i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}}serial loop
  // CHECK-NEXT: if clause
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'T'
  // CHECK-NEXT: self clause
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'T'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt
  //
  // Match the instantiation:
  // CHECK: FunctionDecl{{.*}}TemplFunc{{.*}}implicit_instantiation
  // CHECK-NEXT: TemplateArgument type 'InstTy'
  // CHECK-NEXT: RecordType{{.*}} 'InstTy'
  // CHECK-NEXT: CXXRecord{{.*}} 'InstTy'
  // CHECK-NEXT: CompoundStmt
  //
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}}serial loop
  // CHECK-NEXT: self clause
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}}kernels loop
  // CHECK-NEXT: self clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'bool' <FloatingToBoolean>
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'float' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'const float' lvalue Var{{.*}} 'SomeFloat' 'const float'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'InstTy'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}}parallel loop
  // CHECK-NEXT: self clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'bool' <FloatingToBoolean>
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'float' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'const float' lvalue Var{{.*}} 'SomeFloat' 'const float'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'InstTy'
  // CHECK-NEXT: if clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'bool' <FloatingToBoolean>
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'float' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'const float' lvalue Var{{.*}} 'SomeFloat' 'const float'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'InstTy'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}}serial loop
  // CHECK-NEXT: if clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'bool' <FloatingToBoolean>
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'float' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'const float' lvalue Var{{.*}} 'SomeFloat' 'const float'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'InstTy'
  // CHECK-NEXT: self clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'bool' <FloatingToBoolean>
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'float' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'const float' lvalue Var{{.*}} 'SomeFloat' 'const float'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'InstTy'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

}

struct BoolConversion{ operator bool() const;};
struct InstTy {
  using IntTy = int;
  static constexpr float SomeFloat = 5.0;
  static constexpr BoolConversion BC;
};

void Instantiate() {
  TemplFunc<InstTy>();
}
#endif
