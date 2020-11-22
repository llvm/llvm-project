// RUN: %clang_cc1 -std=c++17 -ast-dump %s | FileCheck %s

void f() noexcept;

// CHECK: VarDecl {{.*}} ref 'void (&)()' cinit
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void ()':'void ()' lvalue <NoOp>
// CHECK-NEXT: DeclRefExpr {{.*}} 'void () noexcept' lvalue Function {{.*}} 'f' 'void () noexcept'
void (&ref)() = f;

struct X {
  typedef void (&ref)() noexcept;
  operator ref();
} x;

// CHECK: VarDecl {{.*}} xp 'void (&)()' cinit
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void ()':'void ()' lvalue <NoOp>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void () noexcept':'void () noexcept' lvalue <UserDefinedConversion>
void (&xp)() = x;
