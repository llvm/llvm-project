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
#pragma acc data if( j < f) default(none)
  ;
  // CHECK-NEXT: OpenACCDataConstruct{{.*}}data
  // CHECK-NEXT: if clause
  // CHECK-NEXT: BinaryOperator{{.*}} 'bool' '<'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'float' <IntegralToFloating>
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'int' lvalue ParmVar{{.*}} 'j' 'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'float' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'float' lvalue ParmVar{{.*}} 'f' 'float'
  // CHECK-NEXT: default(none)
  // CHECK-NEXT: NullStmt

}

int Global;

template<typename T>
void TemplFunc() {
  // CHECK: FunctionTemplateDecl{{.*}}TemplFunc
  // CHECK-NEXT: TemplateTypeParmDecl

  // Match the prototype:
  // CHECK-NEXT: FunctionDecl{{.*}}TemplFunc
  // CHECK-NEXT: CompoundStmt

#pragma acc data default(none) if(T::SomeFloat < typename T::IntTy{})
  ;
  // CHECK-NEXT: OpenACCDataConstruct{{.*}}data
  // CHECK-NEXT: default(none)
  // CHECK-NEXT: if clause
  // CHECK-NEXT: BinaryOperator{{.*}} '<dependent type>' '<'
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'T'
  // CHECK-NEXT: CXXUnresolvedConstructExpr{{.*}} 'typename T::IntTy' 'typename T::IntTy'
  // CHECK-NEXT: InitListExpr{{.*}} 'void'
  // CHECK-NEXT: NullStmt

#pragma acc enter data copyin(Global) if(typename T::IntTy{})
  ;
  // CHECK-NEXT: OpenACCEnterDataConstruct{{.*}}enter data
  // CHECK-NEXT: copyin clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'Global' 'int'
  // CHECK-NEXT: if clause
  // CHECK-NEXT: CXXUnresolvedConstructExpr{{.*}} 'typename T::IntTy' 'typename T::IntTy'
  // CHECK-NEXT: InitListExpr{{.*}} 'void'
  // CHECK-NEXT: NullStmt

#pragma acc exit data copyout(Global) if(T::SomeFloat)
  ;
  // CHECK-NEXT: OpenACCExitDataConstruct{{.*}}exit data
  // CHECK-NEXT: copyout clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'Global' 'int'
  // CHECK-NEXT: if clause
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'T'
  // CHECK-NEXT: NullStmt

#pragma acc host_data use_device(Global) if(T::BC)
  ;
  // CHECK-NEXT: OpenACCHostDataConstruct{{.*}}host_data
  // CHECK-NEXT: use_device clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'Global' 'int'
  // CHECK-NEXT: if clause
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'T'
  // CHECK-NEXT: NullStmt

  // Match the instantiation:
  // CHECK: FunctionDecl{{.*}}TemplFunc{{.*}}implicit_instantiation
  // CHECK-NEXT: TemplateArgument type 'InstTy'
  // CHECK-NEXT: RecordType{{.*}} 'InstTy'
  // CHECK-NEXT: CXXRecord{{.*}} 'InstTy'
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCDataConstruct{{.*}}data
  // CHECK-NEXT: default(none)
  // CHECK-NEXT: if clause
  // CHECK-NEXT: BinaryOperator{{.*}} 'bool' '<'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'float' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'const float' lvalue Var{{.*}} 'SomeFloat' 'const float'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'InstTy'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'float' <IntegralToFloating>
  // CHECK-NEXT: CXXFunctionalCastExpr{{.*}}'typename InstTy::IntTy':'int' functional cast to typename struct InstTy::IntTy <NoOp>
  // CHECK-NEXT: InitListExpr {{.*}}'typename InstTy::IntTy':'int'
  // CHECK-NEXT: NullStmt

  // CHECK-NEXT: OpenACCEnterDataConstruct{{.*}}enter data
  // CHECK-NEXT: copyin clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'Global' 'int'
  // CHECK-NEXT: if clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'bool' <IntegralToBoolean>
  // CHECK-NEXT: CXXFunctionalCastExpr{{.*}}'typename InstTy::IntTy':'int' functional cast to typename struct InstTy::IntTy <NoOp>
  // CHECK-NEXT: InitListExpr {{.*}}'typename InstTy::IntTy':'int'
  // CHECK-NEXT: NullStmt

  // CHECK-NEXT: OpenACCExitDataConstruct{{.*}}exit data
  // CHECK-NEXT: copyout clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'Global' 'int'
  // CHECK-NEXT: if clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'bool' <FloatingToBoolean>
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'float' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'const float' lvalue Var{{.*}} 'SomeFloat' 'const float'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'InstTy'
  // CHECK-NEXT: NullStmt

  // CHECK-NEXT: OpenACCHostDataConstruct{{.*}}host_data
  // CHECK-NEXT: use_device clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'Global' 'int'
  // CHECK-NEXT: if clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'bool' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}} 'bool'
  // CHECK-NEXT: MemberExpr{{.*}} .operator bool
  // CHECK-NEXT: DeclRefExpr{{.*}} 'const BoolConversion' lvalue Var{{.*}} 'BC' 'const BoolConversion'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'InstTy'
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
