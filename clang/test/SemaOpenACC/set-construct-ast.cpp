// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -fopenacc -emit-pch -o %t %s
// RUN: %clang_cc1 %s -fopenacc -include-pch %t -ast-dump-all | FileCheck %s

#ifndef PCH_HELPER
#define PCH_HELPER

int some_int();
long some_long();
void NormalFunc() {
  // CHECK-LABEL: NormalFunc
  // CHECK-NEXT: CompoundStmt

#pragma acc set default_async(some_int()) device_num(some_long()) device_type(DT) if (some_int() < some_long())
  // CHECK-NEXT: OpenACCSetConstruct{{.*}}set
  // CHECK-NEXT: default_async clause
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_int' 'int ()'
  // CHECK-NEXT: device_num clause
  // CHECK-NEXT: CallExpr{{.*}} 'long'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_long' 'long ()'
  // CHECK-NEXT: device_type(DT)
  // CHECK-NEXT: if clause
  // CHECK-NEXT: BinaryOperator{{.*}}'bool' '<'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'long'
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_int' 'int ()'
  // CHECK-NEXT: CallExpr{{.*}} 'long'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_long' 'long ()'
}

template<typename T>
void TemplFunc(T t) {
  // CHECK-LABEL: FunctionTemplateDecl {{.*}}TemplFunc
  // CHECK-NEXT: TemplateTypeParmDecl
  // CHECK-NEXT: FunctionDecl{{.*}}TemplFunc
  // CHECK-NEXT: ParmVarDecl{{.*}} t 'T'
  // CHECK-NEXT: CompoundStmt

#pragma acc set default_async(T::value) device_num(t) device_type(DT) if (T::value < t)
  // CHECK-NEXT: OpenACCSetConstruct{{.*}}set
  // CHECK-NEXT: default_async clause
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'T'
  // CHECK-NEXT: device_num clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'t' 'T'
  // CHECK-NEXT: device_type(DT)
  // CHECK-NEXT: if clause
  // CHECK-NEXT: BinaryOperator{{.*}}'<dependent type>' '<'
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'T'
  // CHECK-NEXT: DeclRefExpr{{.*}}'t' 'T'

  // Instantiation:
  // CHECK-NEXT: FunctionDecl{{.*}} TemplFunc 'void (SomeStruct)' implicit_instantiation
  // CHECK-NEXT: TemplateArgument type 'SomeStruct'
  // CHECK-NEXT: RecordType{{.*}} 'SomeStruct'
  // CHECK-NEXT: CXXRecord{{.*}} 'SomeStruct'
  // CHECK-NEXT: ParmVarDecl{{.*}} t 'SomeStruct'
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCSetConstruct{{.*}}set
  // CHECK-NEXT: default_async clause
  // CHECK-NEXT: ImplicitCastExpr {{.*}}'unsigned int'
  // CHECK-NEXT: DeclRefExpr{{.*}}'value' 'const unsigned int'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'SomeStruct'
  // CHECK-NEXT: device_num clause
  // CHECK-NEXT: ImplicitCastExpr {{.*}}'unsigned int'
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'unsigned int'
  // CHECK-NEXT: MemberExpr{{.*}}.operator unsigned int
  // CHECK-NEXT: DeclRefExpr{{.*}}'t' 'SomeStruct'
  // CHECK-NEXT: device_type(DT)
  // CHECK-NEXT: if clause
  // CHECK-NEXT: BinaryOperator{{.*}}'bool' '<'
  // CHECK-NEXT: ImplicitCastExpr {{.*}}'unsigned int'
  // CHECK-NEXT: DeclRefExpr{{.*}}'value' 'const unsigned int'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'SomeStruct'
  // CHECK-NEXT: ImplicitCastExpr {{.*}}'unsigned int'
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'unsigned int'
  // CHECK-NEXT: MemberExpr{{.*}}.operator unsigned int
  // CHECk-NEXT: DeclRefExpr{{.*}}'t' 'SomeStruct'
}

struct SomeStruct{
  static constexpr unsigned value = 5;
  operator unsigned();
};

void use() {
  TemplFunc(SomeStruct{});
}
#endif
