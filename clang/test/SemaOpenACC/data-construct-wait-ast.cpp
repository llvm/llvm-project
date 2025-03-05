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

  int I;
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl

#pragma acc data copyin(I) wait
  ;
  // CHECK-NEXT: OpenACCDataConstruct{{.*}}data
  // CHECK-NEXT: copyin clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'I' 'int'
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: NullStmt
#pragma acc enter data copyin(I) wait()
  // CHECK: OpenACCEnterDataConstruct{{.*}}enter data
  // CHECK-NEXT: copyin clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'I' 'int'
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: <<<NULL>>>
#pragma acc exit data copyout(I) wait(some_int(), some_long())
  // CHECK: OpenACCExitDataConstruct{{.*}}exit data
  // CHECK-NEXT: copyout clause
  // CHECK-NEXT: DeclRefExpr{{.*}} 'I' 'int'
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int ()' lvalue Function{{.*}} 'some_int' 'int ()'
  // CHECK-NEXT: CallExpr{{.*}}'long'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'long (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'long ()' lvalue Function{{.*}} 'some_long' 'long ()'
#pragma acc data copyin(I) wait(queues:some_int(), some_long())
  ;
  // CHECK: OpenACCDataConstruct{{.*}}data
  // CHECK-NEXT: copyin clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'I' 'int'
  // CHECK-NEXT: wait clause has queues tag
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int ()' lvalue Function{{.*}} 'some_int' 'int ()'
  // CHECK-NEXT: CallExpr{{.*}}'long'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'long (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'long ()' lvalue Function{{.*}} 'some_long' 'long ()'
  // CHECK-NEXT: NullStmt
#pragma acc enter data copyin(I) wait(devnum: some_int() :some_int(), some_long())
  // CHECK: OpenACCEnterDataConstruct{{.*}}enter data
  // CHECK-NEXT: copyin clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'I' 'int'
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
#pragma acc exit data copyout(I) wait(devnum: some_int() : queues :some_int(), some_long()) wait(devnum: some_int() : queues :some_int(), some_long())
  // CHECK: OpenACCExitDataConstruct{{.*}}exit data
  // CHECK-NEXT: copyout clause
  // CHECK-NEXT: DeclRefExpr{{.*}} 'I' 'int'
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
}

template<typename U>
void TemplUses(U u) {
  // CHECK: FunctionTemplateDecl
  // CHECK-NEXT: TemplateTypeParmDecl{{.*}}typename depth 0 index 0 U
  // CHECK-NEXT: FunctionDecl{{.*}} TemplUses 'void (U)'
  // CHECK-NEXT: ParmVarDecl{{.*}} referenced u 'U'
  // CHECK-NEXT: CompoundStmt

  U I;
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl

#pragma acc data copyin(I) wait
  ;
  // CHECK: OpenACCDataConstruct{{.*}}data
  // CHECK-NEXT: copyin clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'I' 'U'
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: NullStmt

#pragma acc enter data copyin(I) wait()
  // CHECK: OpenACCEnterDataConstruct{{.*}}enter data
  // CHECK-NEXT: copyin clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'I' 'U'
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: <<<NULL>>>

#pragma acc exit data copyout(I) wait(U::value, u)
  // CHECK: OpenACCExitDataConstruct{{.*}}exit data
  // CHECK-NEXT: copyout clause
  // CHECK-NEXT: DeclRefExpr{{.*}} 'I' 'U'
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'U'
  // CHECK-NEXT: DeclRefExpr{{.*}} 'U' lvalue ParmVar{{.*}} 'u' 'U'

#pragma acc data copyin(I) wait(queues: U::value, u)
  ;
  // CHECK: OpenACCDataConstruct{{.*}}data
  // CHECK-NEXT: copyin clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'I' 'U'
  // CHECK-NEXT: wait clause has queues tag
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'U'
  // CHECK-NEXT: DeclRefExpr{{.*}} 'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: NullStmt

#pragma acc enter data copyin(I) wait(devnum:u:queues: U::value, u)
  // CHECK: OpenACCEnterDataConstruct{{.*}}data
  // CHECK-NEXT: copyin clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'I' 'U'
  // CHECK-NEXT: wait clause has devnum has queues tag
  // CHECK-NEXT: DeclRefExpr{{.*}} 'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'U'
  // CHECK-NEXT: DeclRefExpr{{.*}} 'U' lvalue ParmVar{{.*}} 'u' 'U'

#pragma acc exit data copyout(I) wait(devnum:u: U::value, u)
  // CHECK: OpenACCExitDataConstruct{{.*}}exit data
  // CHECK-NEXT: copyout clause
  // CHECK-NEXT: DeclRefExpr{{.*}} 'I' 'U'
  // CHECK-NEXT: wait clause has devnum
  // CHECK-NEXT: DeclRefExpr{{.*}} 'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'U'
  // CHECK-NEXT: DeclRefExpr{{.*}} 'U' lvalue ParmVar{{.*}} 'u' 'U'

  // Check the instantiated versions of the above.
  // CHECK: FunctionDecl{{.*}} used TemplUses 'void (HasInt)' implicit_instantiation
  // CHECK-NEXT: TemplateArgument type 'HasInt'
  // CHECK-NEXT: RecordType{{.*}} 'HasInt'
  // CHECK-NEXT: CXXRecord{{.*}} 'HasInt'
  // CHECK-NEXT: ParmVarDecl{{.*}} used u 'HasInt'
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl

  // CHECK: OpenACCDataConstruct{{.*}}data
  // CHECK-NEXT: copyin clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'I' 'HasInt'
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: NullStmt

  // CHECK: OpenACCEnterDataConstruct{{.*}}enter data
  // CHECK-NEXT: copyin clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'I' 'HasInt'
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: <<<NULL>>>

  // CHECK: OpenACCExitDataConstruct{{.*}}exit data
  // CHECK-NEXT: copyout clause
  // CHECK-NEXT: DeclRefExpr{{.*}} 'I' 'HasInt'
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'const int' lvalue Var{{.*}} 'value' 'const int'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'HasInt'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'char' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'char'
  // CHECK-NEXT: MemberExpr{{.*}} '<bound member function type>' .operator char
  // CHECK-NEXT: DeclRefExpr{{.*}} 'HasInt' lvalue ParmVar

  // CHECK: OpenACCDataConstruct{{.*}}data
  // CHECK-NEXT: copyin clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'I' 'HasInt'
  // CHECK-NEXT: wait clause has queues tag
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'const int' lvalue Var{{.*}} 'value' 'const int'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'HasInt'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'char' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'char'
  // CHECK-NEXT: MemberExpr{{.*}} '<bound member function type>' .operator char
  // CHECK-NEXT: DeclRefExpr{{.*}} 'HasInt' lvalue ParmVar
  // CHECK-NEXT: NullStmt

  // CHECK: OpenACCEnterDataConstruct{{.*}}enter data
  // CHECK-NEXT: copyin clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'I' 'HasInt'
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

  // CHECK: OpenACCExitDataConstruct{{.*}}exit data
  // CHECK-NEXT: copyout clause
  // CHECK-NEXT: DeclRefExpr{{.*}} 'I' 'HasInt'
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
