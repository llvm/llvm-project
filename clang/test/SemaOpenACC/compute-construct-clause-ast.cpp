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

#pragma acc kernels if( i < f)
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}kernels
  // CHECK-NEXT: if clause
  // CHECK-NEXT: BinaryOperator{{.*}} 'bool' '<'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'float' <IntegralToFloating>
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int' lvalue ParmVar{{.*}} 'i' 'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'float' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'float' lvalue ParmVar{{.*}} 'f' 'float'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel reduction(+: i)
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: reduction clause Operator: +
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int' lvalue ParmVar{{.*}} 'i' 'int'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc serial reduction(*: f)
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}serial
  // CHECK-NEXT: reduction clause Operator: *
  // CHECK-NEXT: DeclRefExpr{{.*}} 'float' lvalue ParmVar{{.*}} 'f' 'float'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel reduction(max: i)
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: reduction clause Operator: max
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int' lvalue ParmVar{{.*}} 'i' 'int'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc serial reduction(min: f)
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}serial
  // CHECK-NEXT: reduction clause Operator: min
  // CHECK-NEXT: DeclRefExpr{{.*}} 'float' lvalue ParmVar{{.*}} 'f' 'float'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel reduction(&: i)
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: reduction clause Operator: &
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int' lvalue ParmVar{{.*}} 'i' 'int'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc serial reduction(|: f)
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}serial
  // CHECK-NEXT: reduction clause Operator: |
  // CHECK-NEXT: DeclRefExpr{{.*}} 'float' lvalue ParmVar{{.*}} 'f' 'float'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt


#pragma acc parallel reduction(^: i)
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: reduction clause Operator: ^
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int' lvalue ParmVar{{.*}} 'i' 'int'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc serial reduction(&&: f)
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}serial
  // CHECK-NEXT: reduction clause Operator: &&
  // CHECK-NEXT: DeclRefExpr{{.*}} 'float' lvalue ParmVar{{.*}} 'f' 'float'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt


#pragma acc parallel reduction(||: i)
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: reduction clause Operator: ||
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int' lvalue ParmVar{{.*}} 'i' 'int'
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

#pragma acc parallel if(T::SomeFloat < typename T::IntTy{})
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: if clause
  // CHECK-NEXT: BinaryOperator{{.*}} '<dependent type>' '<'
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'T'
  // CHECK-NEXT: CXXUnresolvedConstructExpr{{.*}} 'typename T::IntTy' 'typename T::IntTy'
  // CHECK-NEXT: InitListExpr{{.*}} 'void'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc serial if(typename T::IntTy{})
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}serial
  // CHECK-NEXT: if clause
  // CHECK-NEXT: CXXUnresolvedConstructExpr{{.*}} 'typename T::IntTy' 'typename T::IntTy'
  // CHECK-NEXT: InitListExpr{{.*}} 'void'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc kernels if(T::SomeFloat)
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}kernels
  // CHECK-NEXT: if clause
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'T'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel if(T::BC)
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: if clause
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'T'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc serial self
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}serial
  // CHECK-NEXT: self clause
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc kernels self(T::SomeFloat)
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}kernels
  // CHECK-NEXT: self clause
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'T'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel self(T::SomeFloat) if (T::SomeFloat)
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: self clause
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'T'
  // CHECK-NEXT: if clause
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'T'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc serial if(T::SomeFloat) self(T::SomeFloat)
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}serial
  // CHECK-NEXT: if clause
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'T'
  // CHECK-NEXT: self clause
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'T'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

  T t;
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} t 'T'

#pragma acc parallel reduction(+: t)
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: reduction clause Operator: +
  // CHECK-NEXT: DeclRefExpr{{.*}} 'T' lvalue Var{{.*}} 't' 'T'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc serial reduction(*: T::SomeFloat)
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}serial
  // CHECK-NEXT: reduction clause Operator: *
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'T'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

  typename T::IntTy i;
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} i 'typename T::IntTy'

#pragma acc parallel reduction(max: i)
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: reduction clause Operator: max
  // CHECK-NEXT: DeclRefExpr{{.*}} 'typename T::IntTy' lvalue Var{{.*}} 'i' 'typename T::IntTy'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc serial reduction(min: t)
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}serial
  // CHECK-NEXT: reduction clause Operator: min
  // CHECK-NEXT: DeclRefExpr{{.*}} 'T' lvalue Var{{.*}} 't' 'T'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel reduction(&: T::SomeFloat)
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: reduction clause Operator: &
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'T'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc serial reduction(|: i)
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}serial
  // CHECK-NEXT: reduction clause Operator: |
  // CHECK-NEXT: DeclRefExpr{{.*}} 'typename T::IntTy' lvalue Var{{.*}} 'i' 'typename T::IntTy'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel reduction(^: t)
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: reduction clause Operator: ^
  // CHECK-NEXT: DeclRefExpr{{.*}} 'T' lvalue Var{{.*}} 't' 'T'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc serial reduction(&&: T::SomeFloat)
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}serial
  // CHECK-NEXT: reduction clause Operator: &&
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'T'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel reduction(||: i)
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: reduction clause Operator: ||
  // CHECK-NEXT: DeclRefExpr{{.*}} 'typename T::IntTy' lvalue Var{{.*}} 'i' 'typename T::IntTy'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

  // Match the instantiation:
  // CHECK: FunctionDecl{{.*}}TemplFunc{{.*}}implicit_instantiation
  // CHECK-NEXT: TemplateArgument type 'InstTy'
  // CHECK-NEXT: RecordType{{.*}} 'InstTy'
  // CHECK-NEXT: CXXRecord{{.*}} 'InstTy'
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

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: if clause
  // CHECK-NEXT: BinaryOperator{{.*}} 'bool' '<'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'float' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'const float' lvalue Var{{.*}} 'SomeFloat' 'const float'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'InstTy'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'float' <IntegralToFloating>
  // CHECK-NEXT: CXXFunctionalCastExpr{{.*}}'typename InstTy::IntTy':'int' functional cast to typename InstTy::IntTy <NoOp>
  // CHECK-NEXT: InitListExpr {{.*}}'typename InstTy::IntTy':'int'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}serial
  // CHECK-NEXT: if clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'bool' <IntegralToBoolean>
  // CHECK-NEXT: CXXFunctionalCastExpr{{.*}}'typename InstTy::IntTy':'int' functional cast to typename InstTy::IntTy <NoOp>
  // CHECK-NEXT: InitListExpr {{.*}}'typename InstTy::IntTy':'int'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}kernels
  // CHECK-NEXT: if clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'bool' <FloatingToBoolean>
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'float' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'const float' lvalue Var{{.*}} 'SomeFloat' 'const float'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'InstTy'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: if clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'bool' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}} 'bool'
  // CHECK-NEXT: MemberExpr{{.*}} .operator bool
  // CHECK-NEXT: DeclRefExpr{{.*}} 'const BoolConversion' lvalue Var{{.*}} 'BC' 'const BoolConversion'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'InstTy'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}serial
  // CHECK-NEXT: self clause
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}kernels
  // CHECK-NEXT: self clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'bool' <FloatingToBoolean>
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'float' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'const float' lvalue Var{{.*}} 'SomeFloat' 'const float'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'InstTy'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
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
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}serial
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
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} t 'InstTy'
  // CHECK-NEXT: CXXConstructExpr

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: reduction clause Operator: +
  // CHECK-NEXT: DeclRefExpr{{.*}} 'InstTy' lvalue Var{{.*}} 't' 'InstTy'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}serial
  // CHECK-NEXT: reduction clause Operator: *
  // CHECK-NEXT: DeclRefExpr{{.*}} 'const float' lvalue Var{{.*}} 'SomeFloat' 'const float'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'InstTy'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} i 'typename InstTy::IntTy':'int'

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: reduction clause Operator: max
  // CHECK-NEXT: DeclRefExpr{{.*}} 'typename InstTy::IntTy':'int' lvalue Var{{.*}} 'i' 'typename InstTy::IntTy':'int'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}serial
  // CHECK-NEXT: reduction clause Operator: min
  // CHECK-NEXT: DeclRefExpr{{.*}} 'InstTy' lvalue Var{{.*}} 't' 'InstTy'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: reduction clause Operator: &
  // CHECK-NEXT: DeclRefExpr{{.*}} 'const float' lvalue Var{{.*}} 'SomeFloat' 'const float'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'InstTy'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}serial
  // CHECK-NEXT: reduction clause Operator: |
  // CHECK-NEXT: DeclRefExpr{{.*}} 'typename InstTy::IntTy':'int' lvalue Var{{.*}} 'i' 'typename InstTy::IntTy':'int'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: reduction clause Operator: ^
  // CHECK-NEXT: DeclRefExpr{{.*}} 'InstTy' lvalue Var{{.*}} 't' 'InstTy'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}serial
  // CHECK-NEXT: reduction clause Operator: &&
  // CHECK-NEXT: DeclRefExpr{{.*}} 'const float' lvalue Var{{.*}} 'SomeFloat' 'const float'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'InstTy'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: reduction clause Operator: ||
  // CHECK-NEXT: DeclRefExpr{{.*}} 'typename InstTy::IntTy':'int' lvalue Var{{.*}} 'i' 'typename InstTy::IntTy':'int'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

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
