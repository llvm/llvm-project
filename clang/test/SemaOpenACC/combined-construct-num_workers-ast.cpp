// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -fopenacc -emit-pch -o %t %s
// RUN: %clang_cc1 %s -fopenacc -include-pch %t -ast-dump-all | FileCheck %s

#ifndef PCH_HELPER
#define PCH_HELPER
int some_int();
short some_short();
long some_long();
enum E{};
E some_enum();
struct CorrectConvert {
  operator int();
} Convert;


void NormalUses() {
  // CHECK: FunctionDecl{{.*}}NormalUses
  // CHECK-NEXT: CompoundStmt
#pragma acc parallel loop num_workers(some_int())
  for (unsigned i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} parallel loop
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int ()' lvalue Function{{.*}} 'some_int' 'int ()'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc kernels loop num_workers(some_short())
  for (unsigned i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} kernels loop
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: CallExpr{{.*}}'short'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'short (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'short ()' lvalue Function{{.*}} 'some_short' 'short ()'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc parallel loop num_workers(some_long())
  for (unsigned i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} parallel loop
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: CallExpr{{.*}}'long'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'long (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'long ()' lvalue Function{{.*}} 'some_long' 'long ()'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc parallel loop num_workers(some_enum())
  for (unsigned i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} parallel loop
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: CallExpr{{.*}}'E'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'E (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'E ()' lvalue Function{{.*}} 'some_enum' 'E ()'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc kernels loop num_workers(Convert)
  for (unsigned i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} kernels loop
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'int'
  // CHECK-NEXT: MemberExpr{{.*}} '<bound member function type>' .operator int
  // CHECK-NEXT: DeclRefExpr{{.*}} 'struct CorrectConvert':'CorrectConvert' lvalue Var
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt
}

template<typename T, typename U>
void TemplUses(T t, U u) {
  // CHECK-NEXT: FunctionTemplateDecl
  // CHECK-NEXT: TemplateTypeParmDecl{{.*}}typename depth 0 index 0 T
  // CHECK-NEXT: TemplateTypeParmDecl{{.*}}typename depth 0 index 1 U
  // CHECK-NEXT: FunctionDecl{{.*}} TemplUses 'void (T, U)'
  // CHECK-NEXT: ParmVarDecl{{.*}} referenced t 'T'
  // CHECK-NEXT: ParmVarDecl{{.*}} referenced u 'U'
  // CHECK-NEXT: CompoundStmt

#pragma acc parallel loop num_workers(t)
  for (unsigned i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} parallel loop
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: DeclRefExpr{{.*}} 'T' lvalue ParmVar{{.*}} 't' 'T'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc kernels loop num_workers(u)
  for (unsigned i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} kernels loop
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: DeclRefExpr{{.*}} 'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc parallel loop num_workers(U::value)
  for (unsigned i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} parallel loop
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'U'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc kernels loop num_workers(T{})
  for (unsigned i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} kernels loop
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: CXXUnresolvedConstructExpr{{.*}} 'T' 'T' list
  // CHECK-NEXT: InitListExpr{{.*}} 'void'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc parallel loop num_workers(U{})
  for (unsigned i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} parallel loop
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: CXXUnresolvedConstructExpr{{.*}} 'U' 'U' list
  // CHECK-NEXT: InitListExpr{{.*}} 'void'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc kernels loop num_workers(typename U::IntTy{})
  for (unsigned i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} kernels loop
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: CXXUnresolvedConstructExpr{{.*}} 'typename U::IntTy' 'typename U::IntTy' list
  // CHECK-NEXT: InitListExpr{{.*}} 'void'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc parallel loop num_workers(typename U::ShortTy{})
  for (unsigned i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} parallel loop
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: CXXUnresolvedConstructExpr{{.*}} 'typename U::ShortTy' 'typename U::ShortTy' list
  // CHECK-NEXT: InitListExpr{{.*}} 'void'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

  // Check the instantiated versions of the above.
  // CHECK-NEXT: FunctionDecl{{.*}} used TemplUses 'void (CorrectConvert, HasInt)' implicit_instantiation
  // CHECK-NEXT: TemplateArgument type 'CorrectConvert'
  // CHECK-NEXT: RecordType{{.*}} 'CorrectConvert'
  // CHECK-NEXT: CXXRecord{{.*}} 'CorrectConvert'
  // CHECK-NEXT: TemplateArgument type 'HasInt'
  // CHECK-NEXT: RecordType{{.*}} 'HasInt'
  // CHECK-NEXT: CXXRecord{{.*}} 'HasInt'
  // CHECK-NEXT: ParmVarDecl{{.*}} used t 'CorrectConvert'
  // CHECK-NEXT: ParmVarDecl{{.*}} used u 'HasInt'
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} parallel loop
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'int'
  // CHECK-NEXT: MemberExpr{{.*}} '<bound member function type>' .operator int
  // CHECK-NEXT: DeclRefExpr{{.*}} 'CorrectConvert' lvalue ParmVar
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} kernels loop
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'char' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'char'
  // CHECK-NEXT: MemberExpr{{.*}} '<bound member function type>' .operator char
  // CHECK-NEXT: DeclRefExpr{{.*}} 'HasInt' lvalue ParmVar
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} parallel loop
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'const int' lvalue Var{{.*}} 'value' 'const int'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'HasInt'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} kernels loop
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'int'
  // CHECK-NEXT: MemberExpr{{.*}} '<bound member function type>' .operator int
  // CHECK-NEXT: MaterializeTemporaryExpr{{.*}} 'CorrectConvert' lvalue
  // CHECK-NEXT: CXXFunctionalCastExpr{{.*}} 'CorrectConvert' functional cast to struct CorrectConvert <NoOp>
  // CHECK-NEXT: InitListExpr{{.*}}'CorrectConvert'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} parallel loop
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'char' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'char'
  // CHECK-NEXT: MemberExpr{{.*}} '<bound member function type>' .operator char
  // CHECK-NEXT: MaterializeTemporaryExpr{{.*}} 'HasInt' lvalue
  // CHECK-NEXT: CXXFunctionalCastExpr{{.*}} 'HasInt' functional cast to struct HasInt <NoOp>
  // CHECK-NEXT: InitListExpr{{.*}}'HasInt'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} kernels loop
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: CXXFunctionalCastExpr{{.*}} 'typename HasInt::IntTy':'int' functional cast to typename struct HasInt::IntTy <NoOp>
  // CHECK-NEXT: InitListExpr{{.*}}'typename HasInt::IntTy':'int'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} parallel loop
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: CXXFunctionalCastExpr{{.*}} 'typename HasInt::ShortTy':'short' functional cast to typename struct HasInt::ShortTy <NoOp>
  // CHECK-NEXT: InitListExpr{{.*}}'typename HasInt::ShortTy':'short'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt
}
struct HasInt {
  using IntTy = int;
  using ShortTy = short;
  static constexpr int value = 1;

  operator char();
};

void Inst() {
  TemplUses<CorrectConvert, HasInt>({}, {});
}
#endif // PCH_HELPER
