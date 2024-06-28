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

#pragma acc parallel num_workers(some_int())
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int ()' lvalue Function{{.*}} 'some_int' 'int ()'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

#pragma acc kernels num_workers(some_short())
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}kernels
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: CallExpr{{.*}}'short'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'short (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'short ()' lvalue Function{{.*}} 'some_short' 'short ()'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

#pragma acc parallel num_workers(some_long())
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: CallExpr{{.*}}'long'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'long (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'long ()' lvalue Function{{.*}} 'some_long' 'long ()'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

#pragma acc parallel num_workers(some_enum())
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: CallExpr{{.*}}'E'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'E (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'E ()' lvalue Function{{.*}} 'some_enum' 'E ()'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

#pragma acc kernels num_workers(Convert)
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}kernels
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'int'
  // CHECK-NEXT: MemberExpr{{.*}} '<bound member function type>' .operator int
  // CHECK-NEXT: DeclRefExpr{{.*}} 'struct CorrectConvert':'CorrectConvert' lvalue Var
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

#pragma acc kernels vector_length(some_short())
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}kernels
  // CHECK-NEXT: vector_length clause
  // CHECK-NEXT: CallExpr{{.*}}'short'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'short (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'short ()' lvalue Function{{.*}} 'some_short' 'short ()'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

#pragma acc parallel num_gangs(some_int(), some_long(), some_short())
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: num_gangs clause
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int ()' lvalue Function{{.*}} 'some_int' 'int ()'
  // CHECK-NEXT: CallExpr{{.*}}'long'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'long (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'long ()' lvalue Function{{.*}} 'some_long' 'long ()'
  // CHECK-NEXT: CallExpr{{.*}}'short'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'short (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'short ()' lvalue Function{{.*}} 'some_short' 'short ()'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

#pragma acc kernels num_gangs(some_int())
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}kernels
  // CHECK-NEXT: num_gangs clause
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int ()' lvalue Function{{.*}} 'some_int' 'int ()'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

#pragma acc kernels async(some_int())
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}kernels
  // CHECK-NEXT: async clause
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int ()' lvalue Function{{.*}} 'some_int' 'int ()'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

#pragma acc kernels async
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}kernels
  // CHECK-NEXT: async clause
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

#pragma acc parallel wait
  while (true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt
#pragma acc parallel wait()
  while (true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt
#pragma acc parallel wait(some_int(), some_long())
  while (true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int ()' lvalue Function{{.*}} 'some_int' 'int ()'
  // CHECK-NEXT: CallExpr{{.*}}'long'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'long (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'long ()' lvalue Function{{.*}} 'some_long' 'long ()'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt
#pragma acc parallel wait(queues:some_int(), some_long())
  while (true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: wait clause has queues tag
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int ()' lvalue Function{{.*}} 'some_int' 'int ()'
  // CHECK-NEXT: CallExpr{{.*}}'long'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'long (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'long ()' lvalue Function{{.*}} 'some_long' 'long ()'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt
#pragma acc parallel wait(devnum: some_int() :some_int(), some_long())
  while (true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
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
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt
#pragma acc parallel wait(devnum: some_int() : queues :some_int(), some_long()) wait(devnum: some_int() : queues :some_int(), some_long())
  while (true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
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
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt
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

#pragma acc parallel num_workers(t)
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: DeclRefExpr{{.*}} 'T' lvalue ParmVar{{.*}} 't' 'T'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

#pragma acc kernels num_workers(u)
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}kernels
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: DeclRefExpr{{.*}} 'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

#pragma acc parallel num_workers(U::value)
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'U'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

#pragma acc kernels num_workers(T{})
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}kernels
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: CXXUnresolvedConstructExpr{{.*}} 'T' 'T' list
  // CHECK-NEXT: InitListExpr{{.*}} 'void'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

#pragma acc parallel num_workers(U{})
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: CXXUnresolvedConstructExpr{{.*}} 'U' 'U' list
  // CHECK-NEXT: InitListExpr{{.*}} 'void'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

#pragma acc kernels num_workers(typename U::IntTy{})
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}kernels
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: CXXUnresolvedConstructExpr{{.*}} 'typename U::IntTy' 'typename U::IntTy' list
  // CHECK-NEXT: InitListExpr{{.*}} 'void'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

#pragma acc parallel num_workers(typename U::ShortTy{})
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: CXXUnresolvedConstructExpr{{.*}} 'typename U::ShortTy' 'typename U::ShortTy' list
  // CHECK-NEXT: InitListExpr{{.*}} 'void'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

#pragma acc kernels vector_length(u)
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}kernels
  // CHECK-NEXT: vector_length clause
  // CHECK-NEXT: DeclRefExpr{{.*}} 'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

#pragma acc parallel vector_length(U::value)
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: vector_length clause
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'U'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

#pragma acc kernels num_gangs(u)
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}kernels
  // CHECK-NEXT: num_gangs clause
  // CHECK-NEXT: DeclRefExpr{{.*}} 'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

#pragma acc parallel num_gangs(u, U::value)
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: num_gangs clause
  // CHECK-NEXT: DeclRefExpr{{.*}} 'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'U'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

#pragma acc kernels async
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}kernels
  // CHECK-NEXT: async clause
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

#pragma acc kernels async(u)
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}kernels
  // CHECK-NEXT: async clause
  // CHECK-NEXT: DeclRefExpr{{.*}} 'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

#pragma acc parallel async (U::value)
  while(true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: async clause
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'U'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

#pragma acc parallel wait
  while (true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

#pragma acc parallel wait()
  while (true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

#pragma acc parallel wait(U::value, u)
  while (true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'U'
  // CHECK-NEXT: DeclRefExpr{{.*}} 'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

#pragma acc parallel wait(queues: U::value, u)
  while (true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: wait clause has queues tag
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'U'
  // CHECK-NEXT: DeclRefExpr{{.*}} 'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

#pragma acc parallel wait(devnum:u:queues: U::value, u)
  while (true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: wait clause has devnum has queues tag
  // CHECK-NEXT: DeclRefExpr{{.*}} 'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'U'
  // CHECK-NEXT: DeclRefExpr{{.*}} 'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

#pragma acc parallel wait(devnum:u: U::value, u)
  while (true){}
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: wait clause has devnum
  // CHECK-NEXT: DeclRefExpr{{.*}} 'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'U'
  // CHECK-NEXT: DeclRefExpr{{.*}} 'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt


  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}}EndMarker
  int EndMarker;

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

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'int'
  // CHECK-NEXT: MemberExpr{{.*}} '<bound member function type>' .operator int
  // CHECK-NEXT: DeclRefExpr{{.*}} 'CorrectConvert' lvalue ParmVar
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}kernels
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'char' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'char'
  // CHECK-NEXT: MemberExpr{{.*}} '<bound member function type>' .operator char
  // CHECK-NEXT: DeclRefExpr{{.*}} 'HasInt' lvalue ParmVar
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'const int' lvalue Var{{.*}} 'value' 'const int'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'HasInt'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}kernels
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'int'
  // CHECK-NEXT: MemberExpr{{.*}} '<bound member function type>' .operator int
  // CHECK-NEXT: MaterializeTemporaryExpr{{.*}} 'CorrectConvert' lvalue
  // CHECK-NEXT: CXXFunctionalCastExpr{{.*}} 'CorrectConvert' functional cast to struct CorrectConvert <NoOp>
  // CHECK-NEXT: InitListExpr{{.*}}'CorrectConvert'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: ExprWithCleanups
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'char' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'char'
  // CHECK-NEXT: MemberExpr{{.*}} '<bound member function type>' .operator char
  // CHECK-NEXT: MaterializeTemporaryExpr{{.*}} 'HasInt' lvalue
  // CHECK-NEXT: CXXFunctionalCastExpr{{.*}} 'HasInt' functional cast to struct HasInt <NoOp>
  // CHECK-NEXT: InitListExpr{{.*}}'HasInt'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: ExprWithCleanups
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}kernels
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: CXXFunctionalCastExpr{{.*}} 'typename HasInt::IntTy':'int' functional cast to typename struct HasInt::IntTy <NoOp>
  // CHECK-NEXT: InitListExpr{{.*}}'typename HasInt::IntTy':'int'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: num_workers clause
  // CHECK-NEXT: CXXFunctionalCastExpr{{.*}} 'typename HasInt::ShortTy':'short' functional cast to typename struct HasInt::ShortTy <NoOp>
  // CHECK-NEXT: InitListExpr{{.*}}'typename HasInt::ShortTy':'short'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}kernels
  // CHECK-NEXT: vector_length clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'char' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'char'
  // CHECK-NEXT: MemberExpr{{.*}} '<bound member function type>' .operator char
  // CHECK-NEXT: DeclRefExpr{{.*}} 'HasInt' lvalue ParmVar
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: vector_length clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'const int' lvalue Var{{.*}} 'value' 'const int'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'HasInt'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}kernels
  // CHECK-NEXT: num_gangs clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'char' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'char'
  // CHECK-NEXT: MemberExpr{{.*}} '<bound member function type>' .operator char
  // CHECK-NEXT: DeclRefExpr{{.*}} 'HasInt' lvalue ParmVar
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: num_gangs clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'char' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'char'
  // CHECK-NEXT: MemberExpr{{.*}} '<bound member function type>' .operator char
  // CHECK-NEXT: DeclRefExpr{{.*}} 'HasInt' lvalue ParmVar
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'const int' lvalue Var{{.*}} 'value' 'const int'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'HasInt'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}kernels
  // CHECK-NEXT: async clause
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}kernels
  // CHECK-NEXT: async clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'char' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'char'
  // CHECK-NEXT: MemberExpr{{.*}} '<bound member function type>' .operator char
  // CHECK-NEXT: DeclRefExpr{{.*}} 'HasInt' lvalue ParmVar
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: async clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'const int' lvalue Var{{.*}} 'value' 'const int'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'HasInt'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: wait clause
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'const int' lvalue Var{{.*}} 'value' 'const int'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'HasInt'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'char' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'char'
  // CHECK-NEXT: MemberExpr{{.*}} '<bound member function type>' .operator char
  // CHECK-NEXT: DeclRefExpr{{.*}} 'HasInt' lvalue ParmVar
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
  // CHECK-NEXT: wait clause has queues tag
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'const int' lvalue Var{{.*}} 'value' 'const int'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'HasInt'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'char' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'char'
  // CHECK-NEXT: MemberExpr{{.*}} '<bound member function type>' .operator char
  // CHECK-NEXT: DeclRefExpr{{.*}} 'HasInt' lvalue ParmVar
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
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
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCComputeConstruct{{.*}}parallel
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
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}}EndMarker
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
