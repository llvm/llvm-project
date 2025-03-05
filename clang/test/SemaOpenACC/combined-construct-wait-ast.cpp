// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -fopenacc -emit-pch -o %t %s
// RUN: %clang_cc1 %s -fopenacc -include-pch %t -ast-dump-all | FileCheck %s

#ifndef PCH_HELPER
#define PCH_HELPER

int some_int();
long some_long();

void NormalUses() {
  // CHECK: FunctionDecl{{.*}}NormalUses
  // CHECK-NEXT: CompoundStmt

#pragma acc parallel loop wait
  for (int i = 0; i < 5; ++i) {}
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}}parallel loop
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: ForStmt
#pragma acc serial loop wait()
  for (int i = 0; i < 5; ++i) {}
  // CHECK: OpenACCCombinedConstruct{{.*}}serial loop
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: ForStmt
#pragma acc kernels loop wait(some_int(), some_long())
  for (int i = 0; i < 5; ++i) {}
  // CHECK: OpenACCCombinedConstruct{{.*}}kernels loop
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int ()' lvalue Function{{.*}} 'some_int' 'int ()'
  // CHECK-NEXT: CallExpr{{.*}}'long'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'long (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'long ()' lvalue Function{{.*}} 'some_long' 'long ()'
  // CHECK-NEXT: ForStmt
#pragma acc parallel loop wait(queues:some_int(), some_long())
  for (int i = 0; i < 5; ++i) {}
  // CHECK: OpenACCCombinedConstruct{{.*}}parallel loop
  // CHECK-NEXT: wait clause has queues tag
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int ()' lvalue Function{{.*}} 'some_int' 'int ()'
  // CHECK-NEXT: CallExpr{{.*}}'long'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'long (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'long ()' lvalue Function{{.*}} 'some_long' 'long ()'
  // CHECK-NEXT: ForStmt
#pragma acc serial loop wait(devnum: some_int() :some_int(), some_long())
  for (int i = 0; i < 5; ++i) {}
  // CHECK: OpenACCCombinedConstruct{{.*}}serial loop
  // CHECK-NEXT: wait clause has devnum
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int ()' lvalue Function{{.*}} 'some_int' 'int ()'
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int ()' lvalue Function{{.*}} 'some_int' 'int ()'
  // CHECK-NEXT: CallExpr{{.*}}'long'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'long (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'long ()' lvalue Function{{.*}} 'some_long' 'long ()'
  // CHECK-NEXT: ForStmt
#pragma acc kernels loop wait(devnum: some_int() : queues :some_int(), some_long()) wait(devnum: some_int() : queues :some_int(), some_long())
  for (int i = 0; i < 5; ++i) {}
  // CHECK: OpenACCCombinedConstruct{{.*}}kernels loop
  // CHECK-NEXT: wait clause has devnum has queues tag
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int ()' lvalue Function{{.*}} 'some_int' 'int ()'
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int ()' lvalue Function{{.*}} 'some_int' 'int ()'
  // CHECK-NEXT: CallExpr{{.*}}'long'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'long (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'long ()' lvalue Function{{.*}} 'some_long' 'long ()'
  // CHECK-NEXT: wait clause has devnum has queues tag
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int ()' lvalue Function{{.*}} 'some_int' 'int ()'
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int ()' lvalue Function{{.*}} 'some_int' 'int ()'
  // CHECK-NEXT: CallExpr{{.*}}'long'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'long (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'long ()' lvalue Function{{.*}} 'some_long' 'long ()'
  // CHECK-NEXT: ForStmt
}

template<typename U>
void TemplUses(U u) {
  // CHECK: FunctionTemplateDecl
  // CHECK-NEXT: TemplateTypeParmDecl{{.*}}typename depth 0 index 0 U
  // CHECK-NEXT: FunctionDecl{{.*}} TemplUses 'void (U)'
  // CHECK-NEXT: ParmVarDecl{{.*}} referenced u 'U'
  // CHECK-NEXT: CompoundStmt

#pragma acc parallel loop wait
  for (int i = 0; i < 5; ++i) {}
  // CHECK: OpenACCCombinedConstruct{{.*}}parallel loop
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: ForStmt

#pragma acc serial loop wait()
  for (int i = 0; i < 5; ++i) {}
  // CHECK: OpenACCCombinedConstruct{{.*}}serial loop
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: ForStmt

#pragma acc kernels loop wait(U::value, u)
  for (int i = 0; i < 5; ++i) {}
  // CHECK: OpenACCCombinedConstruct{{.*}}kernels loop
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'U'
  // CHECK-NEXT: DeclRefExpr{{.*}} 'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: ForStmt

#pragma acc parallel loop wait(queues: U::value, u)
  for (int i = 0; i < 5; ++i) {}
  // CHECK: OpenACCCombinedConstruct{{.*}}parallel loop
  // CHECK-NEXT: wait clause has queues tag
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'U'
  // CHECK-NEXT: DeclRefExpr{{.*}} 'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: ForStmt

#pragma acc serial loop wait(devnum:u:queues: U::value, u)
  for (int i = 0; i < 5; ++i) {}
  // CHECK: OpenACCCombinedConstruct{{.*}}serial loop
  // CHECK-NEXT: wait clause has devnum has queues tag
  // CHECK-NEXT: DeclRefExpr{{.*}} 'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'U'
  // CHECK-NEXT: DeclRefExpr{{.*}} 'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: ForStmt

#pragma acc kernels loop wait(devnum:u: U::value, u)
  for (int i = 0; i < 5; ++i) {}
  // CHECK: OpenACCCombinedConstruct{{.*}}kernels loop
  // CHECK-NEXT: wait clause has devnum
  // CHECK-NEXT: DeclRefExpr{{.*}} 'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'U'
  // CHECK-NEXT: DeclRefExpr{{.*}} 'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: ForStmt

  // Check the instantiated versions of the above.
  // CHECK: FunctionDecl{{.*}} used TemplUses 'void (HasInt)' implicit_instantiation
  // CHECK-NEXT: TemplateArgument type 'HasInt'
  // CHECK-NEXT: RecordType{{.*}} 'HasInt'
  // CHECK-NEXT: CXXRecord{{.*}} 'HasInt'
  // CHECK-NEXT: ParmVarDecl{{.*}} used u 'HasInt'
  // CHECK-NEXT: CompoundStmt

  // CHECK: OpenACCCombinedConstruct{{.*}}parallel loop
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: ForStmt

  // CHECK: OpenACCCombinedConstruct{{.*}}serial loop
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: ForStmt

  // CHECK: OpenACCCombinedConstruct{{.*}}kernels loop
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'const int' lvalue Var{{.*}} 'value' 'const int'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'HasInt'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'char' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'char'
  // CHECK-NEXT: MemberExpr{{.*}} '<bound member function type>' .operator char
  // CHECK-NEXT: DeclRefExpr{{.*}} 'HasInt' lvalue ParmVar
  // CHECK-NEXT: ForStmt

  // CHECK: OpenACCCombinedConstruct{{.*}}parallel loop
  // CHECK-NEXT: wait clause has queues tag
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'const int' lvalue Var{{.*}} 'value' 'const int'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'HasInt'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'char' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'char'
  // CHECK-NEXT: MemberExpr{{.*}} '<bound member function type>' .operator char
  // CHECK-NEXT: DeclRefExpr{{.*}} 'HasInt' lvalue ParmVar
  // CHECK-NEXT: ForStmt

  // CHECK: OpenACCCombinedConstruct{{.*}}serial loop
  // CHECK-NEXT: wait clause has devnum has queues tag
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'char' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'char'
  // CHECK-NEXT: MemberExpr{{.*}} '<bound member function type>' .operator char
  // CHECK-NEXT: DeclRefExpr{{.*}} 'HasInt' lvalue ParmVar
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'const int' lvalue Var{{.*}} 'value' 'const int'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'HasInt'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'char' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'char'
  // CHECK-NEXT: MemberExpr{{.*}} '<bound member function type>' .operator char
  // CHECK-NEXT: DeclRefExpr{{.*}} 'HasInt' lvalue ParmVar
  // CHECK-NEXT: ForStmt

  // CHECK: OpenACCCombinedConstruct{{.*}}kernels loop
  // CHECK-NEXT: wait clause has devnum
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'char' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'char'
  // CHECK-NEXT: MemberExpr{{.*}} '<bound member function type>' .operator char
  // CHECK-NEXT: DeclRefExpr{{.*}} 'HasInt' lvalue ParmVar
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'const int' lvalue Var{{.*}} 'value' 'const int'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'HasInt'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'char' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'char'
  // CHECK-NEXT: MemberExpr{{.*}} '<bound member function type>' .operator char
  // CHECK-NEXT: DeclRefExpr{{.*}} 'HasInt' lvalue ParmVar
  // CHECK-NEXT: ForStmt
}

struct HasInt {
  using IntTy = int;
  using ShortTy = short;
  static constexpr int value = 1;

  operator char();
};

void Inst() {
  TemplUses<HasInt>({});
}
#endif
