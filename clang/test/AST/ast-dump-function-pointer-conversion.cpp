// RUN: %clang_cc1 -std=c++17 -ast-dump %s | FileCheck --check-prefixes=CHECK,CXX17 %s
// RUN: %clang_cc1 -std=c++11 -ast-dump %s | FileCheck %s

void f() noexcept;

struct S {
  void m() noexcept;
};

// CHECK: FunctionDecl {{.*}} testFunctionPointerConversion 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} fp 'void (*)()' cinit
// CXX17-NEXT: ImplicitCastExpr {{.*}} 'void (*)()' <FunctionPointerConversion>
// CHECK-NEXT: UnaryOperator {{.*}} 'void (*)() noexcept' prefix '&'
// CHECK-NEXT: DeclRefExpr {{.*}} 'void () noexcept' lvalue Function {{.*}} 'f' 'void () noexcept'
void testFunctionPointerConversion() {
  void (*fp)() = &f;
}

// CHECK: FunctionDecl {{.*}} testMemberFunctionPointerConversion 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} mfp 'void (S::*)()' cinit
// CXX17-NEXT: ImplicitCastExpr {{.*}} 'void (S::*)()' <MemberFunctionPointerConversion>
// CHECK-NEXT: UnaryOperator {{.*}} 'void (S::*)() noexcept' prefix '&'
// CHECK-NEXT: DeclRefExpr {{.*}} 'void () noexcept' CXXMethod {{.*}} 'm' 'void () noexcept'
void testMemberFunctionPointerConversion() {
  void (S::*mfp)() = &S::m;
}
