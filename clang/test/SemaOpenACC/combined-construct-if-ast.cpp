// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -fopenacc -emit-pch -o %t %s
// RUN: %clang_cc1 %s -fopenacc -include-pch %t -ast-dump-all | FileCheck %s

#ifndef PCH_HELPER
#define PCH_HELPER
void NormalFunc(int j, float f) {
  // CHECK: FunctionDecl{{.*}}NormalFunc
  // CHECK-NEXT: ParmVarDecl
  // CHECK-NEXT: ParmVarDecl
  // CHECK-NEXT: CompoundStmt
#pragma acc kernels loop if( j < f)
  for (unsigned i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}}kernels loop
  // CHECK-NEXT: if clause
  // CHECK-NEXT: BinaryOperator{{.*}} 'bool' '<'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'float' <IntegralToFloating>
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int' lvalue ParmVar{{.*}} 'j' 'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'float' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'float' lvalue ParmVar{{.*}} 'f' 'float'
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

#pragma acc parallel loop if(T::SomeFloat < typename T::IntTy{})
  for (unsigned i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}}parallel loop
  // CHECK-NEXT: if clause
  // CHECK-NEXT: BinaryOperator{{.*}} '<dependent type>' '<'
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'T'
  // CHECK-NEXT: CXXUnresolvedConstructExpr{{.*}} 'typename T::IntTy' 'typename T::IntTy'
  // CHECK-NEXT: InitListExpr{{.*}} 'void'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc serial loop if(typename T::IntTy{})
  for (unsigned i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}}serial loop
  // CHECK-NEXT: if clause
  // CHECK-NEXT: CXXUnresolvedConstructExpr{{.*}} 'typename T::IntTy' 'typename T::IntTy'
  // CHECK-NEXT: InitListExpr{{.*}} 'void'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc kernels loop if(T::SomeFloat)
  for (unsigned i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}}kernels loop
  // CHECK-NEXT: if clause
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'T'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc parallel loop if(T::BC)
  for (unsigned i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}}parallel loop
  // CHECK-NEXT: if clause
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'T'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

  // Match the instantiation:
  // CHECK: FunctionDecl{{.*}}TemplFunc{{.*}}implicit_instantiation
  // CHECK-NEXT: TemplateArgument type 'InstTy'
  // CHECK-NEXT: RecordType{{.*}} 'InstTy'
  // CHECK-NEXT: CXXRecord{{.*}} 'InstTy'
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}}parallel loop
  // CHECK-NEXT: if clause
  // CHECK-NEXT: BinaryOperator{{.*}} 'bool' '<'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'float' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'const float' lvalue Var{{.*}} 'SomeFloat' 'const float'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'InstTy'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'float' <IntegralToFloating>
  // CHECK-NEXT: CXXFunctionalCastExpr{{.*}}'typename InstTy::IntTy':'int' functional cast to typename struct InstTy::IntTy <NoOp>
  // CHECK-NEXT: InitListExpr {{.*}}'typename InstTy::IntTy':'int'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}}serial loop
  // CHECK-NEXT: if clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'bool' <IntegralToBoolean>
  // CHECK-NEXT: CXXFunctionalCastExpr{{.*}}'typename InstTy::IntTy':'int' functional cast to typename struct InstTy::IntTy <NoOp>
  // CHECK-NEXT: InitListExpr {{.*}}'typename InstTy::IntTy':'int'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}}kernels loop
  // CHECK-NEXT: if clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'bool' <FloatingToBoolean>
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'float' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'const float' lvalue Var{{.*}} 'SomeFloat' 'const float'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'InstTy'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}}parallel loop
  // CHECK-NEXT: if clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'bool' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}} 'bool'
  // CHECK-NEXT: MemberExpr{{.*}} .operator bool
  // CHECK-NEXT: DeclRefExpr{{.*}} 'const BoolConversion' lvalue Var{{.*}} 'BC' 'const BoolConversion'
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
