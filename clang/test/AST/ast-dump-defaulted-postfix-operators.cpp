// RUN: %clang_cc1 -std=c++2d -ast-dump %s | FileCheck %s

// The implicit definition of a defaulted postfix increment or decrement
// operator for a type C is equivalent to:
//   C tmp(c);
//   ++c;
//   return tmp;

struct S {
  int v;
  S &operator++();
  S operator++(int) = default;
};
void use_member(S s) { s++; }

// CHECK:      CXXMethodDecl {{.*}} used constexpr operator++ 'S (int) noexcept(false)' default implicit-inline
// CHECK-NEXT:   ParmVarDecl {{.*}} 'int'
// CHECK-NEXT:   CompoundStmt
// CHECK-NEXT:     DeclStmt
// CHECK-NEXT:       VarDecl {{.*}} used tmp 'S' nrvo callinit
// CHECK-NEXT:         CXXConstructExpr {{.*}} 'S' 'void (const S &) noexcept'
// CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const S' lvalue <NoOp>
// CHECK-NEXT:             UnaryOperator {{.*}} 'S' lvalue prefix '*' cannot overflow
// CHECK-NEXT:               CXXThisExpr {{.*}} 'S *' this
// CHECK-NEXT:     CXXOperatorCallExpr {{.*}} 'S' lvalue '++'
// CHECK-NEXT:       ImplicitCastExpr {{.*}} 'S &(*)()' <FunctionToPointerDecay>
// CHECK-NEXT:         DeclRefExpr {{.*}} 'S &()' lvalue CXXMethod {{.*}} 'operator++' 'S &()'
// CHECK-NEXT:       UnaryOperator {{.*}} 'S' lvalue prefix '*' cannot overflow
// CHECK-NEXT:         CXXThisExpr {{.*}} 'S *' this
// CHECK-NEXT:     ReturnStmt {{.*}} nrvo_candidate(Var {{.*}} 'tmp' 'S')
// CHECK-NEXT:       CXXConstructExpr {{.*}} 'S' 'void (S &&) noexcept'
// CHECK-NEXT:         ImplicitCastExpr {{.*}} 'S' xvalue <NoOp>
// CHECK-NEXT:           DeclRefExpr {{.*}} 'S' lvalue Var {{.*}} 'tmp' 'S'

struct T {
  int v;
  T &operator--();
};
T operator--(T &self, int) = default;
void use_non_member(T t) { t--; }

// CHECK:      FunctionDecl {{.*}} used constexpr operator-- 'T (T &, int) noexcept(false)' default implicit-inline
// CHECK-NEXT:   ParmVarDecl {{.*}} used self 'T &'
// CHECK-NEXT:   ParmVarDecl {{.*}} 'int'
// CHECK-NEXT:   CompoundStmt
// CHECK-NEXT:     DeclStmt
// CHECK-NEXT:       VarDecl {{.*}} used tmp 'T' nrvo callinit
// CHECK-NEXT:         CXXConstructExpr {{.*}} 'T' 'void (const T &) noexcept'
// CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const T' lvalue <NoOp>
// CHECK-NEXT:             DeclRefExpr {{.*}} 'T' lvalue ParmVar {{.*}} 'self' 'T &'
// CHECK-NEXT:     CXXOperatorCallExpr {{.*}} 'T' lvalue '--'
// CHECK-NEXT:       ImplicitCastExpr {{.*}} 'T &(*)()' <FunctionToPointerDecay>
// CHECK-NEXT:         DeclRefExpr {{.*}} 'T &()' lvalue CXXMethod {{.*}} 'operator--' 'T &()'
// CHECK-NEXT:       DeclRefExpr {{.*}} 'T' lvalue ParmVar {{.*}} 'self' 'T &'
// CHECK-NEXT:     ReturnStmt {{.*}} nrvo_candidate(Var {{.*}} 'tmp' 'T')
// CHECK-NEXT:       CXXConstructExpr {{.*}} 'T' 'void (T &&) noexcept'
// CHECK-NEXT:         ImplicitCastExpr {{.*}} 'T' xvalue <NoOp>
// CHECK-NEXT:           DeclRefExpr {{.*}} 'T' lvalue Var {{.*}} 'tmp' 'T'

struct U {
  int v;
  U &operator++();
  U operator++(this U &self, int) = default;
};
void use_explicit_object(U u) { u++; }

// CHECK:      CXXMethodDecl {{.*}} used constexpr operator++ 'U (U &, int) noexcept(false)' default implicit-inline
// CHECK-NEXT:   ParmVarDecl {{.*}} used self this 'U &'
// CHECK-NEXT:   ParmVarDecl {{.*}} 'int'
// CHECK-NEXT:   CompoundStmt
// CHECK-NEXT:     DeclStmt
// CHECK-NEXT:       VarDecl {{.*}} used tmp 'U' nrvo callinit
// CHECK-NEXT:         CXXConstructExpr {{.*}} 'U' 'void (const U &) noexcept'
// CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const U' lvalue <NoOp>
// CHECK-NEXT:             DeclRefExpr {{.*}} 'U' lvalue ParmVar {{.*}} 'self' 'U &'
// CHECK-NEXT:     CXXOperatorCallExpr {{.*}} 'U' lvalue '++'
// CHECK-NEXT:       ImplicitCastExpr {{.*}} 'U &(*)()' <FunctionToPointerDecay>
// CHECK-NEXT:         DeclRefExpr {{.*}} 'U &()' lvalue CXXMethod {{.*}} 'operator++' 'U &()'
// CHECK-NEXT:       DeclRefExpr {{.*}} 'U' lvalue ParmVar {{.*}} 'self' 'U &'
// CHECK-NEXT:     ReturnStmt {{.*}} nrvo_candidate(Var {{.*}} 'tmp' 'U')
// CHECK-NEXT:       CXXConstructExpr {{.*}} 'U' 'void (U &&) noexcept'
// CHECK-NEXT:         ImplicitCastExpr {{.*}} 'U' xvalue <NoOp>
// CHECK-NEXT:           DeclRefExpr {{.*}} 'U' lvalue Var {{.*}} 'tmp' 'U'

// A deleted defaulted postfix operator has no body.
struct Deleted {
  Deleted operator++(int) = default;
};
// CHECK:      CXXRecordDecl {{.*}} struct Deleted definition
// CHECK:      CXXMethodDecl {{.*}} operator++ 'Deleted (int)' default_delete
// CHECK-NEXT:   ParmVarDecl {{.*}} 'int'
// CHECK-NOT:    CompoundStmt
