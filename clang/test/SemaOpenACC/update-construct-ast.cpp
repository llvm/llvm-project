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

#pragma acc update if_present if (some_int() < some_long())
  // CHECK-NEXT: OpenACCUpdateConstruct{{.*}}update
  // CHECK-NEXT: if_present clause
  // CHECK-NEXT: if clause
  // CHECK-NEXT: BinaryOperator{{.*}}'bool' '<'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'long'
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_int' 'int ()'
  // CHECK-NEXT: CallExpr{{.*}} 'long'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_long' 'long ()'

#pragma acc update wait async device_type(A) dtype(B)
  // CHECK-NEXT: OpenACCUpdateConstruct{{.*}}update
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: async clause
  // CHECK-NEXT: device_type(A)
  // CHECK-NEXT: dtype(B)
#pragma acc update wait(some_int(), some_long()) async(some_int())
  // CHECK-NEXT: OpenACCUpdateConstruct{{.*}}update
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_int' 'int ()'
  // CHECK-NEXT: CallExpr{{.*}}'long'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_long' 'long ()'
  // CHECK-NEXT: async clause
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_int' 'int ()'
#pragma acc update wait(queues:some_int(), some_long())
  // CHECK-NEXT: OpenACCUpdateConstruct{{.*}}update
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_int' 'int ()'
  // CHECK-NEXT: CallExpr{{.*}}'long'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_long' 'long ()'
#pragma acc update wait(devnum: some_int() :some_int(), some_long())
  // CHECK-NEXT: OpenACCUpdateConstruct{{.*}}update
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_int' 'int ()'
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_int' 'int ()'
  // CHECK-NEXT: CallExpr{{.*}}'long'
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

#pragma acc update if_present if (T::value < t)
  // CHECK-NEXT: OpenACCUpdateConstruct{{.*}}update
  // CHECK-NEXT: if_present clause
  // CHECK-NEXT: if clause
  // CHECK-NEXT: BinaryOperator{{.*}}'<dependent type>' '<'
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'T'
  // CHECK-NEXT: DeclRefExpr{{.*}}'t' 'T'

#pragma acc update wait async device_type(T) dtype(U)
  // CHECK-NEXT: OpenACCUpdateConstruct{{.*}}update
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: async clause
  // CHECK-NEXT: device_type(T)
  // CHECK-NEXT: dtype(U)
#pragma acc update wait(T::value, t) async(T::value)
  // CHECK-NEXT: OpenACCUpdateConstruct{{.*}}update
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}}'<dependent type>'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'T'
  // CHECK-NEXT: DeclRefExpr{{.*}} 't' 'T'
  // CHECK-NEXT: async clause
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}}'<dependent type>'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'T'
#pragma acc update wait(queues:T::value, t) async(t)
  // CHECK-NEXT: OpenACCUpdateConstruct{{.*}}update
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}}'<dependent type>'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'T'
  // CHECK-NEXT: DeclRefExpr{{.*}} 't' 'T'
  // CHECK-NEXT: async clause
  // CHECK-NEXT: DeclRefExpr{{.*}} 't' 'T'
#pragma acc update wait(devnum: T::value:t, T::value)
  // CHECK-NEXT: OpenACCUpdateConstruct{{.*}}update
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}}'<dependent type>'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'T'
  // CHECK-NEXT: DeclRefExpr{{.*}} 't' 'T'
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}}'<dependent type>'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'T'

  // Instantiation:
  // CHECK-NEXT: FunctionDecl{{.*}} TemplFunc 'void (SomeStruct)' implicit_instantiation
  // CHECK-NEXT: TemplateArgument type 'SomeStruct'
  // CHECK-NEXT: RecordType{{.*}} 'SomeStruct'
  // CHECK-NEXT: CXXRecord{{.*}} 'SomeStruct'
  // CHECK-NEXT: ParmVarDecl{{.*}} t 'SomeStruct'
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCUpdateConstruct{{.*}}update
  // CHECK-NEXT: if_present clause
  // CHECK-NEXT: if clause
  // CHECK-NEXT: BinaryOperator{{.*}}'bool' '<'
  // CHECK-NEXT: ImplicitCastExpr {{.*}}'unsigned int'
  // CHECK-NEXT: DeclRefExpr{{.*}}'value' 'const unsigned int'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'SomeStruct'
  // CHECK-NEXT: ImplicitCastExpr {{.*}}'unsigned int'
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'unsigned int'
  // CHECK-NEXT: MemberExpr{{.*}}.operator unsigned int
  // CHECK-NEXT: DeclRefExpr{{.*}}'t' 'SomeStruct'

  // CHECK-NEXT: OpenACCUpdateConstruct{{.*}}update
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: async clause
  // CHECK-NEXT: device_type(T)
  // CHECK-NEXT: dtype(U)

  // CHECK-NEXT: OpenACCUpdateConstruct{{.*}}update
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'unsigned int'
  // CHECK-NEXT: DeclRefExpr{{.*}}'value' 'const unsigned int'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'SomeStruct'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'unsigned int'
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'unsigned int'
  // CHECK-NEXT: MemberExpr{{.*}}.operator unsigned int
  // CHECK-NEXT: DeclRefExpr{{.*}}'t' 'SomeStruct'
  // CHECK-NEXT: async clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'unsigned int'
  // CHECK-NEXT: DeclRefExpr{{.*}}'value' 'const unsigned int'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'SomeStruct'

  // CHECK-NEXT: OpenACCUpdateConstruct{{.*}}update
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'unsigned int'
  // CHECK-NEXT: DeclRefExpr{{.*}}'value' 'const unsigned int'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'SomeStruct'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'unsigned int'
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'unsigned int'
  // CHECK-NEXT: MemberExpr{{.*}}.operator unsigned int
  // CHECK-NEXT: DeclRefExpr{{.*}}'t' 'SomeStruct'
  // CHECK-NEXT: async clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'unsigned int'
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'unsigned int'
  // CHECK-NEXT: MemberExpr{{.*}}.operator unsigned int
  // CHECK-NEXT: DeclRefExpr{{.*}}'t' 'SomeStruct'

  // CHECK-NEXT: OpenACCUpdateConstruct{{.*}}update
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'unsigned int'
  // CHECK-NEXT: DeclRefExpr{{.*}}'value' 'const unsigned int'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'SomeStruct'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'unsigned int'
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'unsigned int'
  // CHECK-NEXT: MemberExpr{{.*}}.operator unsigned int
  // CHECK-NEXT: DeclRefExpr{{.*}}'t' 'SomeStruct'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'unsigned int'
  // CHECK-NEXT: DeclRefExpr{{.*}}'value' 'const unsigned int'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'SomeStruct'
}

struct SomeStruct{
  static constexpr unsigned value = 5;
  operator unsigned();
};
void use() {
  TemplFunc(SomeStruct{});
}
#endif
