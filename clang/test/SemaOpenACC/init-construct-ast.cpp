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

#pragma acc init
  // CHECK-NEXT: OpenACCInitConstruct{{.*}}init
#pragma acc init if (some_int() < some_long())
  // CHECK-NEXT: OpenACCInitConstruct{{.*}}init
  // CHECK-NEXT: if clause
  // CHECK-NEXT: BinaryOperator{{.*}} 'bool' '<'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'long'
  // CHECK-NEXT: CallExpr
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_int' 'int ()'
  // CHECK-NEXT: CallExpr
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_long' 'long ()'
#pragma acc init device_num(some_int())
  // CHECK-NEXT: OpenACCInitConstruct{{.*}}init
  // CHECK-NEXT: device_num clause
  // CHECK-NEXT: CallExpr
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_int' 'int ()'
#pragma acc init device_type(T)
  // CHECK-NEXT: OpenACCInitConstruct{{.*}}init
  // CHECK-NEXT: device_type(T)
#pragma acc init if (some_int() < some_long()) device_type(T) device_num(some_int())
  // CHECK-NEXT: OpenACCInitConstruct{{.*}}init
  // CHECK-NEXT: if clause
  // CHECK-NEXT: BinaryOperator{{.*}} 'bool' '<'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'long'
  // CHECK-NEXT: CallExpr
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_int' 'int ()'
  // CHECK-NEXT: CallExpr
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_long' 'long ()'
  // CHECK-NEXT: device_type(T)
  // CHECK-NEXT: device_num clause
  // CHECK-NEXT: CallExpr
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_int' 'int ()'
}

template<typename T>
void TemplFunc(T t) {
  // CHECK-LABEL: FunctionTemplateDecl {{.*}}TemplFunc
  // CHECK-NEXT: TemplateTypeParmDecl
  // CHECK-NEXT: FunctionDecl{{.*}}TemplFunc
  // CHECK-NEXT: ParmVarDecl{{.*}} t 'T'
  // CHECK-NEXT: CompoundStmt

#pragma acc init
  // CHECK-NEXT: OpenACCInitConstruct{{.*}}init
#pragma acc init if (T::value > t)
  // CHECK-NEXT: OpenACCInitConstruct{{.*}}init
  // CHECK-NEXT: if clause
  // CHECK-NEXT: BinaryOperator{{.*}} '<dependent type>' '>'
  // CHECK-NEXT: DependentScopeDeclRefExpr
  // CHECK-NEXT: NestedNameSpecifier{{.*}} 'T'
  // CHECK-NEXT: DeclRefExpr{{.*}} 't' 'T'
#pragma acc init device_num(t)
  // CHECK-NEXT: OpenACCInitConstruct{{.*}}init
  // CHECK-NEXT: device_num clause
  // CHECK-NEXT: DeclRefExpr{{.*}} 't' 'T'
#pragma acc init device_type(T)
  // CHECK-NEXT: OpenACCInitConstruct{{.*}}init
  // CHECK-NEXT: device_type(T)
#pragma acc init if (T::value > t) device_type(T) device_num(t)
  // CHECK-NEXT: OpenACCInitConstruct{{.*}}init
  // CHECK-NEXT: if clause
  // CHECK-NEXT: BinaryOperator{{.*}} '<dependent type>' '>'
  // CHECK-NEXT: DependentScopeDeclRefExpr
  // CHECK-NEXT: NestedNameSpecifier{{.*}} 'T'
  // CHECK-NEXT: DeclRefExpr{{.*}} 't' 'T'
  // CHECK-NEXT: device_type(T)
  // CHECK-NEXT: device_num clause
  // CHECK-NEXT: DeclRefExpr{{.*}} 't' 'T'

  // Instantiation:
  // CHECK-NEXT: FunctionDecl{{.*}} TemplFunc 'void (SomeStruct)' implicit_instantiation
  // CHECK-NEXT: TemplateArgument type 'SomeStruct'
  // CHECK-NEXT: RecordType{{.*}} 'SomeStruct'
  // CHECK-NEXT: CXXRecord{{.*}} 'SomeStruct'
  // CHECK-NEXT: ParmVarDecl{{.*}} t 'SomeStruct'
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCInitConstruct{{.*}}init

  // CHECK-NEXT: OpenACCInitConstruct{{.*}}init
  // CHECK-NEXT: if clause
  // CHECK-NEXT: BinaryOperator{{.*}} 'bool' '>'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'unsigned int'
  // CHECK-NEXT: DeclRefExpr{{.*}}'value' 'const unsigned int'
  // CHECK-NEXT: NestedNameSpecifier{{.*}} 'SomeStruct'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'unsigned int'
  // CHECK-NEXT: CXXMemberCallExpr{{.*}} 'unsigned int'
  // CHECK-NEXT: MemberExpr{{.*}} .operator unsigned int
  // CHECK-NEXT: DeclRefExpr{{.*}} 't' 'SomeStruct'

  // CHECK-NEXT: OpenACCInitConstruct{{.*}}init
  // CHECK-NEXT: device_num clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'unsigned int'
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'unsigned int'
  // CHECK-NEXT: MemberExpr{{.*}} .operator unsigned int
  // CHECK-NEXT: DeclRefExpr{{.*}} 't' 'SomeStruct'

  // CHECK-NEXT: OpenACCInitConstruct{{.*}}init
  // CHECK-NEXT: device_type(T)

  // CHECK-NEXT: OpenACCInitConstruct{{.*}}init
  // CHECK-NEXT: if clause
  // CHECK-NEXT: BinaryOperator{{.*}} 'bool' '>'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'unsigned int'
  // CHECK-NEXT: DeclRefExpr{{.*}}'value' 'const unsigned int'
  // CHECK-NEXT: NestedNameSpecifier{{.*}} 'SomeStruct'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'unsigned int'
  // CHECK-NEXT: CXXMemberCallExpr{{.*}} 'unsigned int'
  // CHECK-NEXT: MemberExpr{{.*}} .operator unsigned int
  // CHECK-NEXT: DeclRefExpr{{.*}} 't' 'SomeStruct'
  // CHECK-NEXT: device_type(T)
  // CHECK-NEXT: device_num clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'unsigned int'
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'unsigned int'
  // CHECK-NEXT: MemberExpr{{.*}} .operator unsigned int
  // CHECK-NEXT: DeclRefExpr{{.*}} 't' 'SomeStruct'
}

struct SomeStruct{
  static constexpr unsigned value = 5;
  operator unsigned();
};

void use() {
  TemplFunc(SomeStruct{});
}
#endif
