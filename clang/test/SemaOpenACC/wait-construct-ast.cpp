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

#pragma acc wait async(some_int())
  // CHECK-NEXT: OpenACCWaitConstruct{{.*}}wait
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: async clause
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_int'
#pragma acc wait() async
  // CHECK-NEXT: OpenACCWaitConstruct{{.*}}wait
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: async clause
#pragma acc wait(some_int(), some_long()) if (some_int() < some_long())
  // CHECK-NEXT: OpenACCWaitConstruct{{.*}}wait
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_int'
  // CHECK-NEXT: CallExpr{{.*}}'long'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'long (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_long'
  // CHECK-NEXT: if clause
  // CHECK-NEXT: BinaryOperator{{.*}} 'bool' '<'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'long'
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_int'
  // CHECK-NEXT: CallExpr{{.*}}'long'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'long (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_long'
#pragma acc wait(queues:some_int(), some_long())
  // CHECK-NEXT: OpenACCWaitConstruct{{.*}}wait
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_int'
  // CHECK-NEXT: CallExpr{{.*}}'long'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'long (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_long'
#pragma acc wait(devnum:some_int() : queues:some_int(), some_long())
  // CHECK-NEXT: OpenACCWaitConstruct{{.*}}wait
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_int'
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_int'
  // CHECK-NEXT: CallExpr{{.*}}'long'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'long (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_long'
#pragma acc wait(devnum:some_int() : some_int(), some_long())
  // CHECK-NEXT: OpenACCWaitConstruct{{.*}}wait
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_int'
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_int'
  // CHECK-NEXT: CallExpr{{.*}}'long'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'long (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_long'
}

template<typename T>
void TemplFunc(T t) {
  // CHECK-LABEL: FunctionTemplateDecl {{.*}}TemplFunc
  // CHECK-NEXT: TemplateTypeParmDecl
  // CHECK-NEXT: FunctionDecl{{.*}}TemplFunc
  // CHECK-NEXT: ParmVarDecl{{.*}} t 'T'
  // CHECK-NEXT: CompoundStmt

#pragma acc wait async(T::value)
  // CHECK-NEXT: OpenACCWaitConstruct{{.*}}wait
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: async clause
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>'
  // CHECK-NEXT: NestedNameSpecifier{{.*}} 'T'
#pragma acc wait() async
  // CHECK-NEXT: OpenACCWaitConstruct{{.*}}wait
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: async clause
#pragma acc wait(t, T::value) if (T::value > t)
  // CHECK-NEXT: OpenACCWaitConstruct{{.*}}wait
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'T' lvalue ParmVar{{.*}} 't' 'T'
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>'
  // CHECK-NEXT: NestedNameSpecifier{{.*}} 'T'
  // CHECK-NEXT: if clause
  // CHECK-NEXT: BinaryOperator{{.*}} '<dependent type>' '>'
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>'
  // CHECK-NEXT: NestedNameSpecifier{{.*}} 'T'
  // CHECK-NEXT: DeclRefExpr{{.*}} 'T' lvalue ParmVar{{.*}} 't' 'T'
#pragma acc wait(queues:typename T::IntTy{}, T::value) if (typename T::IntTy{} < typename T::ShortTy{})
  // CHECK-NEXT: OpenACCWaitConstruct{{.*}}wait
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: CXXUnresolvedConstructExpr{{.*}}'typename T::IntTy' list
  // CHECK-NEXT: InitListExpr
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>'
  // CHECK-NEXT: NestedNameSpecifier{{.*}} 'T'
  // CHECK-NEXT: if clause
  // CHECK-NEXT: BinaryOperator{{.*}} '<dependent type>' '<'
  // CHECK-NEXT: CXXUnresolvedConstructExpr{{.*}}'typename T::IntTy' list
  // CHECK-NEXT: InitListExpr
  // CHECK-NEXT: CXXUnresolvedConstructExpr{{.*}}'typename T::ShortTy' list
  // CHECK-NEXT: InitListExpr
#pragma acc wait(devnum:typename T::ShortTy{} : queues:some_int(), T::value)
  // CHECK-NEXT: OpenACCWaitConstruct{{.*}}wait
  // CHECK-NEXT: CXXUnresolvedConstructExpr{{.*}}'typename T::ShortTy' list
  // CHECK-NEXT: InitListExpr
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_int'
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>'
  // CHECK-NEXT: NestedNameSpecifier{{.*}} 'T'
#pragma acc wait(devnum:typename T::ShortTy{} : T::value, some_long())
  // CHECK-NEXT: OpenACCWaitConstruct{{.*}}wait
  // CHECK-NEXT: CXXUnresolvedConstructExpr{{.*}}'typename T::ShortTy' list
  // CHECK-NEXT: InitListExpr
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>'
  // CHECK-NEXT: NestedNameSpecifier{{.*}} 'T'
  // CHECK-NEXT: CallExpr{{.*}}'long'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'long (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_long'

  // Instantiation:
  // CHECK-NEXT: FunctionDecl{{.*}} TemplFunc 'void (HasInt)' implicit_instantiation
  // CHECK-NEXT: TemplateArgument type 'HasInt'
  // CHECK-NEXT: RecordType{{.*}} 'HasInt'
  // CHECK-NEXT: CXXRecord{{.*}} 'HasInt'
  // CHECK-NEXT: ParmVarDecl{{.*}} t 'HasInt'
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCWaitConstruct{{.*}}wait
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: async clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int'
  // CHECK-NEXT: DeclRefExpr{{.*}}'value' 'const int'
  // CHECK-NEXT: NestedNameSpecifier {{.*}}'HasInt'
  //
  // CHECK-NEXT: OpenACCWaitConstruct{{.*}}wait
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: async clause
  //
  // CHECK-NEXT: OpenACCWaitConstruct{{.*}}wait
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'char' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'char'
  // CHECK-NEXT: MemberExpr{{.*}}.operator char
  // CHECK-NEXT: DeclRefExpr{{.*}} 'HasInt' lvalue ParmVar{{.*}} 't' 'HasInt'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int'
  // CHECK-NEXT: DeclRefExpr{{.*}}'value' 'const int'
  // CHECK-NEXT: NestedNameSpecifier {{.*}}'HasInt'
  // CHECK-NEXT: if clause
  // CHECK-NEXT: BinaryOperator{{.*}} 'bool' '>'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int'
  // CHECK-NEXT: DeclRefExpr{{.*}}'value' 'const int'
  // CHECK-NEXT: NestedNameSpecifier {{.*}}'HasInt'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int' <IntegralCast>
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'char' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'char'
  // CHECK-NEXT: MemberExpr{{.*}}.operator char
  // CHECK-NEXT: DeclRefExpr{{.*}} 'HasInt' lvalue ParmVar{{.*}} 't' 'HasInt'
  //
  // CHECK-NEXT: OpenACCWaitConstruct{{.*}}wait
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: CXXFunctionalCastExpr{{.*}}'typename HasInt::IntTy':'int'
  // CHECK-NEXT: InitListExpr{{.*}}'typename HasInt::IntTy':'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int'
  // CHECK-NEXT: DeclRefExpr{{.*}}'value' 'const int'
  // CHECK-NEXT: NestedNameSpecifier {{.*}}'HasInt'
  // CHECK-NEXT: if clause
  // CHECK-NEXT: BinaryOperator{{.*}} 'bool' '<'
  // CHECK-NEXT: CXXFunctionalCastExpr{{.*}}'typename HasInt::IntTy':'int'
  // CHECK-NEXT: InitListExpr{{.*}}'typename HasInt::IntTy':'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int'
  // CHECK-NEXT: CXXFunctionalCastExpr{{.*}}'typename HasInt::ShortTy':'short'
  // CHECK-NEXT: InitListExpr{{.*}}'typename HasInt::ShortTy':'short'
  //
  // CHECK-NEXT: OpenACCWaitConstruct{{.*}}wait
  // CHECK-NEXT: CXXFunctionalCastExpr{{.*}}'typename HasInt::ShortTy':'short'
  // CHECK-NEXT: InitListExpr{{.*}}'typename HasInt::ShortTy':'short'
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int'
  // CHECK-NEXT: DeclRefExpr{{.*}}'value' 'const int'
  // CHECK-NEXT: NestedNameSpecifier {{.*}}'HasInt'
  //
  // CHECK-NEXT: OpenACCWaitConstruct{{.*}}wait
  // CHECK-NEXT: CXXFunctionalCastExpr{{.*}}'typename HasInt::ShortTy':'short'
  // CHECK-NEXT: InitListExpr{{.*}}'typename HasInt::ShortTy':'short'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int'
  // CHECK-NEXT: DeclRefExpr{{.*}}'value' 'const int'
  // CHECK-NEXT: NestedNameSpecifier {{.*}}'HasInt'
  // CHECK-NEXT: CallExpr{{.*}}'long'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'long (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'some_long'
}

struct HasInt {
  using IntTy = int;
  using ShortTy = short;
  static constexpr int value = 1;

  operator char();
};
void use() {
  TemplFunc(HasInt{});
}
#endif
