// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -fopenacc -emit-pch -o %t %s
// RUN: %clang_cc1 %s -fopenacc -include-pch %t -ast-dump-all | FileCheck %s

#ifndef PCH_HELPER
#define PCH_HELPER
int some_int();
short some_short();
long some_long();
struct CorrectConvert {
  operator int();
} Convert;


void NormalUses() {
  // CHECK: FunctionDecl{{.*}}NormalUses
  // CHECK-NEXT: CompoundStmt

#pragma acc parallel loop num_gangs(some_int(), some_long(), some_short())
  for (unsigned i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} parallel loop
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
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc kernels loop num_gangs(some_int())
  for (unsigned i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} kernels loop
  // CHECK-NEXT: num_gangs clause
  // CHECK-NEXT: CallExpr{{.*}}'int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int ()' lvalue Function{{.*}} 'some_int' 'int ()'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt
}

template<typename T, typename U>
void TemplUses(T t, U u) {
  // CHECK-NEXT: FunctionTemplateDecl
  // CHECK-NEXT: TemplateTypeParmDecl{{.*}}typename depth 0 index 0 T
  // CHECK-NEXT: TemplateTypeParmDecl{{.*}}typename depth 0 index 1 U
  // CHECK-NEXT: FunctionDecl{{.*}} TemplUses 'void (T, U)'
  // CHECK-NEXT: ParmVarDecl{{.*}} t 'T'
  // CHECK-NEXT: ParmVarDecl{{.*}} u 'U'
  // CHECK-NEXT: CompoundStmt

#pragma acc kernels loop num_gangs(u)
  for (unsigned i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} kernels loop
  // CHECK-NEXT: num_gangs clause
  // CHECK-NEXT: DeclRefExpr{{.*}} 'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc parallel loop num_gangs(u, U::value)
  for (unsigned i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} parallel loop
  // CHECK-NEXT: num_gangs clause
  // CHECK-NEXT: DeclRefExpr{{.*}} 'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: DependentScopeDeclRefExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'U'
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
  // CHECK-NEXT: ParmVarDecl{{.*}} t 'CorrectConvert'
  // CHECK-NEXT: ParmVarDecl{{.*}} u 'HasInt'
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} kernels loop
  // CHECK-NEXT: num_gangs clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'char' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'char'
  // CHECK-NEXT: MemberExpr{{.*}} '<bound member function type>' .operator char
  // CHECK-NEXT: DeclRefExpr{{.*}} 'HasInt' lvalue ParmVar
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} parallel loop
  // CHECK-NEXT: num_gangs clause
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'char' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'char'
  // CHECK-NEXT: MemberExpr{{.*}} '<bound member function type>' .operator char
  // CHECK-NEXT: DeclRefExpr{{.*}} 'HasInt' lvalue ParmVar
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}} 'const int' lvalue Var{{.*}} 'value' 'const int'
  // CHECK-NEXT: NestedNameSpecifier TypeSpec 'HasInt'
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
