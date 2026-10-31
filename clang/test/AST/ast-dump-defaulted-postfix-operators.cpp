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
// CHECK-NEXT:       VarDecl {{.*}} used 'S' nrvo callinit
// CHECK-NEXT:         CXXConstructExpr {{.*}} 'S' 'void (const S &) noexcept'
// CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const S' lvalue <NoOp>
// CHECK-NEXT:             UnaryOperator {{.*}} 'S' lvalue prefix '*' cannot overflow
// CHECK-NEXT:               CXXThisExpr {{.*}} 'S *' this
// CHECK-NEXT:     CXXOperatorCallExpr {{.*}} 'S' lvalue '++'
// CHECK-NEXT:       ImplicitCastExpr {{.*}} 'S &(*)()' <FunctionToPointerDecay>
// CHECK-NEXT:         DeclRefExpr {{.*}} 'S &()' lvalue CXXMethod {{.*}} 'operator++' 'S &()'
// CHECK-NEXT:       UnaryOperator {{.*}} 'S' lvalue prefix '*' cannot overflow
// CHECK-NEXT:         CXXThisExpr {{.*}} 'S *' this
// CHECK-NEXT:     ReturnStmt {{.*}} nrvo_candidate(Var {{.*}} 'S')
// CHECK-NEXT:       CXXConstructExpr {{.*}} 'S' 'void (S &&) noexcept'
// CHECK-NEXT:         ImplicitCastExpr {{.*}} 'S' xvalue <NoOp>
// CHECK-NEXT:           DeclRefExpr {{.*}} 'S' lvalue Var {{.*}} 'S'

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
// CHECK-NEXT:       VarDecl {{.*}} used 'T' nrvo callinit
// CHECK-NEXT:         CXXConstructExpr {{.*}} 'T' 'void (const T &) noexcept'
// CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const T' lvalue <NoOp>
// CHECK-NEXT:             DeclRefExpr {{.*}} 'T' lvalue ParmVar {{.*}} 'self' 'T &'
// CHECK-NEXT:     CXXOperatorCallExpr {{.*}} 'T' lvalue '--'
// CHECK-NEXT:       ImplicitCastExpr {{.*}} 'T &(*)()' <FunctionToPointerDecay>
// CHECK-NEXT:         DeclRefExpr {{.*}} 'T &()' lvalue CXXMethod {{.*}} 'operator--' 'T &()'
// CHECK-NEXT:       DeclRefExpr {{.*}} 'T' lvalue ParmVar {{.*}} 'self' 'T &'
// CHECK-NEXT:     ReturnStmt {{.*}} nrvo_candidate(Var {{.*}} 'T')
// CHECK-NEXT:       CXXConstructExpr {{.*}} 'T' 'void (T &&) noexcept'
// CHECK-NEXT:         ImplicitCastExpr {{.*}} 'T' xvalue <NoOp>
// CHECK-NEXT:           DeclRefExpr {{.*}} 'T' lvalue Var {{.*}} 'T'

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
// CHECK-NEXT:       VarDecl {{.*}} used 'U' nrvo callinit
// CHECK-NEXT:         CXXConstructExpr {{.*}} 'U' 'void (const U &) noexcept'
// CHECK-NEXT:           ImplicitCastExpr {{.*}} 'const U' lvalue <NoOp>
// CHECK-NEXT:             DeclRefExpr {{.*}} 'U' lvalue ParmVar {{.*}} 'self' 'U &'
// CHECK-NEXT:     CXXOperatorCallExpr {{.*}} 'U' lvalue '++'
// CHECK-NEXT:       ImplicitCastExpr {{.*}} 'U &(*)()' <FunctionToPointerDecay>
// CHECK-NEXT:         DeclRefExpr {{.*}} 'U &()' lvalue CXXMethod {{.*}} 'operator++' 'U &()'
// CHECK-NEXT:       DeclRefExpr {{.*}} 'U' lvalue ParmVar {{.*}} 'self' 'U &'
// CHECK-NEXT:     ReturnStmt {{.*}} nrvo_candidate(Var {{.*}} 'U')
// CHECK-NEXT:       CXXConstructExpr {{.*}} 'U' 'void (U &&) noexcept'
// CHECK-NEXT:         ImplicitCastExpr {{.*}} 'U' xvalue <NoOp>
// CHECK-NEXT:           DeclRefExpr {{.*}} 'U' lvalue Var {{.*}} 'U'

// A deleted defaulted postfix operator has no body.
struct Deleted {
  Deleted operator++(int) = default;
};
// CHECK:      CXXRecordDecl {{.*}} struct Deleted definition
// CHECK:      CXXMethodDecl {{.*}} operator++ 'Deleted (int)' default_delete
// CHECK-NEXT:   ParmVarDecl {{.*}} 'int'
// CHECK-NOT:    CompoundStmt

// A constrained defaulted postfix operator keeps its requires-clause, and its
// implicit definition calls the prefix operator selected by overload
// resolution, which is the more constrained one.
template <typename T> concept Small = sizeof(T) <= 4;
template <typename T> struct Constrained {
  T v;
  Constrained &operator++() requires Small<T>;
  Constrained &operator++();
  Constrained operator++(int) requires Small<T> = default;
};
void use_constrained(Constrained<int> c) { c++; }

// CHECK:      ClassTemplateSpecializationDecl {{.*}} struct Constrained definition
// CHECK:        CXXMethodDecl [[PREFIX:0x[0-9a-f]+]] {{.*}} used operator++ 'Constrained<int> &()'
// CHECK-NEXT:     ConceptSpecializationExpr {{.*}} 'bool' Concept {{.*}} 'Small'
// CHECK:        CXXMethodDecl {{.*}} operator++ 'Constrained<int> &()'
// CHECK-NEXT:   CXXMethodDecl {{.*}} used constexpr operator++ 'Constrained<int> (int) noexcept(false)' implicit_instantiation default implicit-inline
// CHECK-NEXT:     ParmVarDecl {{.*}} 'int'
// CHECK-NEXT:     ConceptSpecializationExpr {{.*}} 'bool' Concept {{.*}} 'Small'
// CHECK:          CompoundStmt
// CHECK-NEXT:       DeclStmt
// CHECK-NEXT:         VarDecl {{.*}} used 'Constrained<int>' nrvo callinit
// CHECK:            CXXOperatorCallExpr {{.*}} 'Constrained<int>' lvalue '++'
// CHECK-NEXT:         ImplicitCastExpr {{.*}} 'Constrained<int> &(*)()' <FunctionToPointerDecay>
// CHECK-NEXT:           DeclRefExpr {{.*}} 'Constrained<int> &()' lvalue CXXMethod [[PREFIX]] 'operator++' 'Constrained<int> &()'
// CHECK:            ReturnStmt {{.*}} nrvo_candidate(Var {{.*}} 'Constrained<int>')
