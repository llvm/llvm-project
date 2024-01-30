// RUN: %clang_cc1 -std=c++23 -triple x86_64-linux-gnu -fcxx-exceptions -ast-dump %s \
// RUN: | FileCheck -strict-whitespace %s

namespace P2718R0 {

// Test basic
struct A {
  int a[3] = {1, 2, 3};
  A() {}
  ~A() {}
  const int *begin() const { return a; }
  const int *end() const { return a + 3; }
  A& r() { return *this; }
  A g() { return A(); }
};

A g() { return A(); }
const A &f1(const A &t) { return t; }

void test1() {
  [[maybe_unused]] int sum = 0;
  // CHECK: FunctionDecl {{.*}} test1 'void ()'
  // CHECK:      |   `-CXXForRangeStmt {{.*}}
  // CHECK-NEXT: |     |-<<<NULL>>>
  // CHECK-NEXT: |     |-DeclStmt {{.*}}
  // CHECK-NEXT: |     | `-VarDecl {{.*}} implicit used __range1 'const A &' cinit
  // CHECK-NEXT: |     |   `-ExprWithCleanups {{.*}} 'const A':'const P2718R0::A' lvalue
  // CHECK-NEXT: |     |     `-CallExpr {{.*}} 'const A':'const P2718R0::A' lvalue
  // CHECK-NEXT: |     |       |-ImplicitCastExpr {{.*}} 'const A &(*)(const A &)' <FunctionToPointerDecay>
  // CHECK-NEXT: |     |       | `-DeclRefExpr {{.*}} 'const A &(const A &)' lvalue Function {{.*}} 'f1' 'const A &(const A &)'
  // CHECK-NEXT: |     |       `-MaterializeTemporaryExpr {{.*}} 'const A':'const P2718R0::A' lvalue extended by Var {{.*}} '__range1' 'const A &'
  // CHECK-NEXT: |     |         `-ImplicitCastExpr {{.*}} 'const A':'const P2718R0::A' <NoOp>
  // CHECK-NEXT: |     |           `-CXXBindTemporaryExpr {{.*}} 'A':'P2718R0::A' (CXXTemporary {{.*}})
  // CHECK-NEXT: |     |             `-CallExpr {{.*}} 'A':'P2718R0::A'
  // CHECK-NEXT: |     |               `-ImplicitCastExpr {{.*}} 'A (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT: |     |                 `-DeclRefExpr {{.*}} 'A ()' lvalue Function {{.*}} 'g' 'A ()'
  for (auto e : f1(g()))
    sum += e;
}

struct B : A {};
int (&f(const A *))[3];
const A *g(const A &);
void bar(int) {}

void test2() {
  // CHECK: FunctionDecl {{.*}} test2 'void ()'
  // CHECK:      |   `-CXXForRangeStmt {{.*}}
  // CHECK-NEXT: |     |-<<<NULL>>>
  // CHECK-NEXT: |     |-DeclStmt {{.*}}
  // CHECK-NEXT: |     | `-VarDecl {{.*}} implicit used __range1 'int (&)[3]' cinit
  // CHECK-NEXT: |     |   `-ExprWithCleanups {{.*}} 'int[3]' lvalue
  // CHECK-NEXT: |     |     `-CallExpr {{.*}} 'int[3]' lvalue
  // CHECK-NEXT: |     |       |-ImplicitCastExpr {{.*}} 'int (&(*)(const A *))[3]' <FunctionToPointerDecay>
  // CHECK-NEXT: |     |       | `-DeclRefExpr {{.*}} 'int (&(const A *))[3]' lvalue Function {{.*}} 'f' 'int (&(const A *))[3]'
  // CHECK-NEXT: |     |       `-CallExpr {{.*}} 'const A *'
  // CHECK-NEXT: |     |         |-ImplicitCastExpr {{.*}} 'const A *(*)(const A &)' <FunctionToPointerDecay>
  // CHECK-NEXT: |     |         | `-DeclRefExpr {{.*}} 'const A *(const A &)' lvalue Function {{.*}} 'g' 'const A *(const A &)'
  // CHECK-NEXT: |     |         `-ImplicitCastExpr {{.*}} 'const A':'const P2718R0::A' lvalue <DerivedToBase (A)>
  // CHECK-NEXT: |     |           `-MaterializeTemporaryExpr {{.*}} 'const B':'const P2718R0::B' lvalue extended by Var {{.*}} '__range1' 'int (&)[3]'
  // CHECK-NEXT: |     |             `-ImplicitCastExpr {{.*}} 'const B':'const P2718R0::B' <NoOp>
  // CHECK-NEXT: |     |               `-CXXBindTemporaryExpr {{.*}} 'B':'P2718R0::B' (CXXTemporary {{.*}})
  // CHECK-NEXT: |     |                 `-CXXTemporaryObjectExpr {{.*}} 'B':'P2718R0::B' 'void () noexcept(false)' zeroing
  for (auto e : f(g(B())))
    bar(e);
}

// Test discard statement.
struct LockGuard {
    LockGuard() {}
    ~LockGuard() {}
};

void test3() {
  int v[] = {42, 17, 13};

  // CHECK: FunctionDecl {{.*}} test3 'void ()'
  // CHECK:      -CXXForRangeStmt {{.*}}
  // CHECK-NEXT:  |-<<<NULL>>>
  // CHECK-NEXT:  |-DeclStmt {{.*}}
  // CHECK-NEXT:  | `-VarDecl {{.*}} implicit used __range1 'int (&)[3]' cinit
  // CHECK-NEXT:  |   `-ExprWithCleanups {{.*}} 'int[3]' lvalue
  // CHECK-NEXT:  |     `-BinaryOperator {{.*}} 'int[3]' lvalue ','
  // CHECK-NEXT:  |       |-CXXStaticCastExpr {{.*}} 'void' static_cast<void> <ToVoid>
  // CHECK-NEXT:  |       | `-MaterializeTemporaryExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' xvalue extended by Var {{.*}} '__range1' 'int (&)[3]'
  // CHECK-NEXT:  |       |   `-CXXBindTemporaryExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' (CXXTemporary {{.*}})
  // CHECK-NEXT:  |       |     `-CXXTemporaryObjectExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' 'void ()'
  // CHECK-NEXT:  |       `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} 'v' 'int[3]'
  for ([[maybe_unused]] int x : static_cast<void>(LockGuard()), v)
    LockGuard guard;

  // CHECK:      -CXXForRangeStmt {{.*}}
  // CHECK-NEXT:  |-<<<NULL>>>
  // CHECK-NEXT:  |-DeclStmt {{.*}}
  // CHECK-NEXT:  | `-VarDecl {{.*}} implicit used __range1 'int (&)[3]' cinit
  // CHECK-NEXT:  |   `-ExprWithCleanups {{.*}} 'int[3]' lvalue
  // CHECK-NEXT:  |     `-BinaryOperator {{.*}} 'int[3]' lvalue ','
  // CHECK-NEXT:  |       |-CStyleCastExpr {{.*}} 'void' <ToVoid>
  // CHECK-NEXT:  |       | `-MaterializeTemporaryExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' xvalue extended by Var {{.*}} '__range1' 'int (&)[3]'
  // CHECK-NEXT:  |       |   `-CXXBindTemporaryExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' (CXXTemporary {{.*}})
  // CHECK-NEXT:  |       |     `-CXXTemporaryObjectExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' 'void ()'
  // CHECK-NEXT:  |       `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} 'v' 'int[3]'
  for ([[maybe_unused]] int x : (void)LockGuard(), v)
    LockGuard guard;

  // CHECK:      -CXXForRangeStmt {{.*}}
  // CHECK-NEXT:  |-<<<NULL>>>
  // CHECK-NEXT:  |-DeclStmt {{.*}}
  // CHECK-NEXT:  | `-VarDecl {{.*}} implicit used __range1 'int (&)[3]' cinit
  // CHECK-NEXT:  |   `-ExprWithCleanups {{.*}} 'int[3]' lvalue
  // CHECK-NEXT:  |     `-BinaryOperator {{.*}} 'int[3]' lvalue ','
  // CHECK-NEXT:  |       |-MaterializeTemporaryExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' xvalue extended by Var {{.*}} '__range1' 'int (&)[3]'
  // CHECK-NEXT:  |       | `-CXXBindTemporaryExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' (CXXTemporary {{.*}})
  // CHECK-NEXT:  |       |   `-CXXTemporaryObjectExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' 'void ()'
  // CHECK-NEXT:  |       `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} 'v' 'int[3]'
  for ([[maybe_unused]] int x : LockGuard(), v)
    LockGuard guard;
}

// Test default arg
int (&default_arg_fn(const A & = A()))[3];
void test4() {

  // CHECK: FunctionDecl {{.*}} test4 'void ()'
  // FIXME: Should dump CXXDefaultArgExpr->getExpr() if CXXDefaultArgExpr has been rewrited?
  for (auto e : default_arg_fn()) 
    bar(e);
}

struct DefaultA {
  DefaultA() {}
  ~DefaultA() {}
};

A foo(const A&, const DefaultA &Default = DefaultA()) {
  return A();
}

void test5() {
  for (auto e : default_arg_fn(foo(foo(foo(A())))))
    bar(e);
}

struct C : public A {
  C() {}
  C(int, const C &, const DefaultA & = DefaultA()) {}
};

void test6() {
  for (auto e : C(0, C(0, C(0, C()))))
    bar(e);
}

// Test member call
void test7() {
  // CHECK: FunctionDecl {{.*}} test7 'void ()'
  // CHECK:      -CXXForRangeStmt {{.*}}
  // CHECK-NEXT:  |-<<<NULL>>>
  // CHECK-NEXT:  |-DeclStmt {{.*}}
  // CHECK-NEXT:  | `-VarDecl {{.*}} implicit used __range1 'A &&' cinit
  // CHECK-NEXT:  |   `-ExprWithCleanups {{.*}} 'A':'P2718R0::A' xvalue
  // CHECK-NEXT:  |     `-MaterializeTemporaryExpr {{.*}} 'A':'P2718R0::A' xvalue extended by Var {{.*}} '__range1' 'A &&'
  // CHECK-NEXT:  |       `-CXXBindTemporaryExpr {{.*}} 'A':'P2718R0::A' (CXXTemporary {{.*}})
  // CHECK-NEXT:  |         `-CXXMemberCallExpr {{.*}} 'A':'P2718R0::A'
  // CHECK-NEXT:  |           `-MemberExpr {{.*}} '<bound member function type>' .g {{.*}}
  // CHECK-NEXT:  |             `-CXXMemberCallExpr {{.*}} 'A':'P2718R0::A' lvalue
  // CHECK-NEXT:  |               `-MemberExpr {{.*}} '<bound member function type>' .r {{.*}}
  // CHECK-NEXT:  |                 `-MaterializeTemporaryExpr {{.*}} 'A':'P2718R0::A' xvalue extended by Var {{.*}} '__range1' 'A &&'
  // CHECK-NEXT:  |                   `-CXXBindTemporaryExpr {{.*}} 'A':'P2718R0::A' (CXXTemporary {{.*}})
  // CHECK-NEXT:  |                     `-CXXMemberCallExpr {{.*}} 'A':'P2718R0::A'
  // CHECK-NEXT:  |                       `-MemberExpr {{.*}} '<bound member function type>' .g {{.*}}
  // CHECK-NEXT:  |                         `-CXXMemberCallExpr {{.*}} 'A':'P2718R0::A' lvalue
  // CHECK-NEXT:  |                           `-MemberExpr {{.*}} '<bound member function type>' .r {{.*}}
  // CHECK-NEXT:  |                             `-MaterializeTemporaryExpr {{.*}} 'A':'P2718R0::A' xvalue extended by Var {{.*}} '__range1' 'A &&'
  // CHECK-NEXT:  |                               `-CXXBindTemporaryExpr {{.*}} 'A':'P2718R0::A' (CXXTemporary {{.*}})
  // CHECK-NEXT:  |                                 `-CXXMemberCallExpr {{.*}} 'A':'P2718R0::A'
  // CHECK-NEXT:  |                                   `-MemberExpr {{.*}} '<bound member function type>' .g {{.*}}
  // CHECK-NEXT:  |                                     `-CXXMemberCallExpr {{.*}} 'A':'P2718R0::A' lvalue
  // CHECK-NEXT:  |                                       `-MemberExpr {{.*}} '<bound member function type>' .r {{.*}}
  // CHECK-NEXT:  |                                         `-MaterializeTemporaryExpr {{.*}} 'A':'P2718R0::A' xvalue extended by Var {{.*}} '__range1' 'A &&'
  // CHECK-NEXT:  |                                           `-CXXBindTemporaryExpr {{.*}} 'A':'P2718R0::A' (CXXTemporary {{.*}})
  // CHECK-NEXT:  |                                             `-CallExpr {{.*}} 'A':'P2718R0::A'
  // CHECK-NEXT:  |                                               `-ImplicitCastExpr {{.*}} 'A (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT:  |                                                 `-DeclRefExpr {{.*}} 'A ()' lvalue Function {{.*}} 'g' 'A ()'
  for (auto e : g().r().g().r().g().r().g())
    bar(e);
}

// Test basic && dependent context
template <typename T> T dg() { return T(); }
template <typename T> const T &df1(const T &t) { return t; }

void test8() {
  [[maybe_unused]] int sum = 0;
  // CHECK: FunctionDecl {{.*}} test8 'void ()'
  // CHECK:      -CXXForRangeStmt {{.*}}
  // CHECK-NEXT:  |-<<<NULL>>>
  // CHECK-NEXT:  |-DeclStmt {{.*}}
  // CHECK-NEXT:  | `-VarDecl {{.*}} implicit used __range1 'const P2718R0::A &' cinit
  // CHECK-NEXT:  |   `-ExprWithCleanups {{.*}} 'const P2718R0::A' lvalue
  // CHECK-NEXT:  |     `-CallExpr {{.*}} 'const P2718R0::A' lvalue
  // CHECK-NEXT:  |       |-ImplicitCastExpr {{.*}} 'const P2718R0::A &(*)(const P2718R0::A &)' <FunctionToPointerDecay>
  // CHECK-NEXT:  |       | `-DeclRefExpr {{.*}} 'const P2718R0::A &(const P2718R0::A &)' lvalue Function {{.*}} 'df1' 'const P2718R0::A &(const P2718R0::A &)' (FunctionTemplate {{.*}} 'df1')
  // CHECK-NEXT:  |       `-MaterializeTemporaryExpr {{.*}} 'const P2718R0::A' lvalue extended by Var {{.*}} '__range1' 'const P2718R0::A &'
  // CHECK-NEXT:  |         `-ImplicitCastExpr {{.*}} 'const P2718R0::A' <NoOp>
  // CHECK-NEXT:  |           `-CXXBindTemporaryExpr {{.*}} 'P2718R0::A' (CXXTemporary {{.*}})
  // CHECK-NEXT:  |             `-CallExpr {{.*}} 'P2718R0::A'
  // CHECK-NEXT:  |               `-ImplicitCastExpr {{.*}} 'P2718R0::A (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT:  |                 `-DeclRefExpr {{.*}} 'P2718R0::A ()' lvalue Function {{.*}} 'dg' 'P2718R0::A ()' (FunctionTemplate {{.*}} 'dg')
  for (auto e : df1(dg<A>()))
    sum += e;
}

template <typename T> int (&df2(const T *))[3];
const A *dg2(const A &);

void test9() {
  // CHECK: FunctionDecl {{.*}} test9 'void ()'
  // CHECK:      -CXXForRangeStmt {{.*}}
  // CHECK-NEXT:  |-<<<NULL>>>
  // CHECK-NEXT:  |-DeclStmt {{.*}}
  // CHECK-NEXT:  | `-VarDecl {{.*}} implicit used __range1 'int (&)[3]' cinit
  // CHECK-NEXT:  |   `-ExprWithCleanups {{.*}} 'int[3]' lvalue
  // CHECK-NEXT:  |     `-CallExpr {{.*}} 'int[3]' lvalue
  // CHECK-NEXT:  |       |-ImplicitCastExpr {{.*}} 'int (&(*)(const P2718R0::A *))[3]' <FunctionToPointerDecay>
  // CHECK-NEXT:  |       | `-DeclRefExpr {{.*}} 'int (&(const P2718R0::A *))[3]' lvalue Function {{.*}} 'df2' 'int (&(const P2718R0::A *))[3]' (FunctionTemplate {{.*}} 'df2')
  // CHECK-NEXT:  |       `-CallExpr {{.*}} 'const A *'
  // CHECK-NEXT:  |         |-ImplicitCastExpr {{.*}} 'const A *(*)(const A &)' <FunctionToPointerDecay>
  // CHECK-NEXT:  |         | `-DeclRefExpr {{.*}} 'const A *(const A &)' lvalue Function {{.*}} 'dg2' 'const A *(const A &)'
  // CHECK-NEXT:  |         `-ImplicitCastExpr {{.*}} 'const A':'const P2718R0::A' lvalue <DerivedToBase (A)>
  // CHECK-NEXT:  |           `-MaterializeTemporaryExpr {{.*}} 'const B':'const P2718R0::B' lvalue extended by Var {{.*}} '__range1' 'int (&)[3]'
  // CHECK-NEXT:  |             `-ImplicitCastExpr {{.*}} 'const B':'const P2718R0::B' <NoOp>
  // CHECK-NEXT:  |               `-CXXBindTemporaryExpr {{.*}} 'B':'P2718R0::B' (CXXTemporary {{.*}})
  // CHECK-NEXT:  |                 `-CXXTemporaryObjectExpr {{.*}} 'B':'P2718R0::B' 'void () noexcept(false)' zeroing
  for (auto e : df2(dg2(B())))
    bar(e);
}

// Test discard statement && dependent context
void test10() {
  int v[] = {42, 17, 13};

  // CHECK: FunctionDecl {{.*}} test10 'void ()'
  // CHECK:      -CXXForRangeStmt {{.*}}
  // CHECK-NEXT:  |-<<<NULL>>>
  // CHECK-NEXT:  |-DeclStmt {{.*}}
  // CHECK-NEXT:  | `-VarDecl {{.*}} implicit used __range1 'int (&)[3]' cinit
  // CHECK-NEXT:  |   `-ExprWithCleanups {{.*}} 'int[3]' lvalue
  // CHECK-NEXT:  |     `-BinaryOperator {{.*}} 'int[3]' lvalue ','
  // CHECK-NEXT:  |       |-CXXStaticCastExpr {{.*}} 'void' static_cast<void> <ToVoid>
  // CHECK-NEXT:  |       | `-CallExpr {{.*}} 'const P2718R0::LockGuard' lvalue
  // CHECK-NEXT:  |       |   |-ImplicitCastExpr {{.*}} 'const P2718R0::LockGuard &(*)(const P2718R0::LockGuard &)' <FunctionToPointerDecay>
  // CHECK-NEXT:  |       |   | `-DeclRefExpr {{.*}} 'const P2718R0::LockGuard &(const P2718R0::LockGuard &)' lvalue Function {{.*}} 'df1' 'const P2718R0::LockGuard &(const P2718R0::LockGuard &)' (FunctionTemplate {{.*}} 'df1')
  // CHECK-NEXT:  |       |   `-MaterializeTemporaryExpr {{.*}} 'const LockGuard':'const P2718R0::LockGuard' lvalue extended by Var {{.*}} '__range1' 'int (&)[3]'
  // CHECK-NEXT:  |       |     `-ImplicitCastExpr {{.*}} 'const LockGuard':'const P2718R0::LockGuard' <NoOp>
  // CHECK-NEXT:  |       |       `-CXXBindTemporaryExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' (CXXTemporary {{.*}})
  // CHECK-NEXT:  |       |         `-CXXTemporaryObjectExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' 'void ()'
  // CHECK-NEXT:  |       `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} 'v' 'int[3]'
  for ([[maybe_unused]] int x : static_cast<void>(df1(LockGuard())), v)
    LockGuard guard;

  // CHECK:      -CXXForRangeStmt {{.*}}
  // CHECK-NEXT:  |-<<<NULL>>>
  // CHECK-NEXT:  |-DeclStmt {{.*}}
  // CHECK-NEXT:  | `-VarDecl {{.*}} implicit used __range1 'int (&)[3]' cinit
  // CHECK-NEXT:  |   `-ExprWithCleanups {{.*}} 'int[3]' lvalue
  // CHECK-NEXT:  |     `-BinaryOperator {{.*}} 'int[3]' lvalue ','
  // CHECK-NEXT:  |       |-CStyleCastExpr {{.*}} 'void' <ToVoid>
  // CHECK-NEXT:  |       | `-CallExpr {{.*}} 'const P2718R0::LockGuard' lvalue
  // CHECK-NEXT:  |       |   |-ImplicitCastExpr {{.*}} 'const P2718R0::LockGuard &(*)(const P2718R0::LockGuard &)' <FunctionToPointerDecay>
  // CHECK-NEXT:  |       |   | `-DeclRefExpr {{.*}} 'const P2718R0::LockGuard &(const P2718R0::LockGuard &)' lvalue Function {{.*}} 'df1' 'const P2718R0::LockGuard &(const P2718R0::LockGuard &)' (FunctionTemplate {{.*}} 'df1')
  // CHECK-NEXT:  |       |   `-MaterializeTemporaryExpr {{.*}} 'const LockGuard':'const P2718R0::LockGuard' lvalue extended by Var {{.*}} '__range1' 'int (&)[3]'
  // CHECK-NEXT:  |       |     `-ImplicitCastExpr {{.*}} 'const LockGuard':'const P2718R0::LockGuard' <NoOp>
  // CHECK-NEXT:  |       |       `-CXXBindTemporaryExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' (CXXTemporary {{.*}})
  // CHECK-NEXT:  |       |         `-CXXTemporaryObjectExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' 'void ()'
  // CHECK-NEXT:  |       `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} 'v' 'int[3]'
  for ([[maybe_unused]] int x : (void)df1(LockGuard()), v)
    LockGuard guard;

  // CHECK:      -CXXForRangeStmt {{.*}}
  // CHECK-NEXT:  |-<<<NULL>>>
  // CHECK-NEXT:  |-DeclStmt {{.*}}
  // CHECK-NEXT:  | `-VarDecl {{.*}} implicit used __range1 'int (&)[3]' cinit
  // CHECK-NEXT:  |   `-ExprWithCleanups {{.*}} 'int[3]' lvalue
  // CHECK-NEXT:  |     `-BinaryOperator {{.*}} 'int[3]' lvalue ','
  // CHECK-NEXT:  |       |-BinaryOperator {{.*}} 'const P2718R0::LockGuard' lvalue ','
  // CHECK-NEXT:  |       | |-CallExpr {{.*}} 'const P2718R0::LockGuard' lvalue
  // CHECK-NEXT:  |       | | |-ImplicitCastExpr {{.*}} 'const P2718R0::LockGuard &(*)(const P2718R0::LockGuard &)' <FunctionToPointerDecay>
  // CHECK-NEXT:  |       | | | `-DeclRefExpr {{.*}} 'const P2718R0::LockGuard &(const P2718R0::LockGuard &)' lvalue Function {{.*}} 'df1' 'const P2718R0::LockGuard &(const P2718R0::LockGuard &)' (FunctionTemplate {{.*}} 'df1')
  // CHECK-NEXT:  |       | | `-MaterializeTemporaryExpr {{.*}} 'const LockGuard':'const P2718R0::LockGuard' lvalue extended by Var {{.*}} '__range1' 'int (&)[3]'
  // CHECK-NEXT:  |       | |   `-ImplicitCastExpr {{.*}} 'const LockGuard':'const P2718R0::LockGuard' <NoOp>
  // CHECK-NEXT:  |       | |     `-CXXBindTemporaryExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' (CXXTemporary {{.*}})
  // CHECK-NEXT:  |       | |       `-CXXTemporaryObjectExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' 'void ()'
  // CHECK-NEXT:  |       | `-CallExpr {{.*}} 'const P2718R0::LockGuard' lvalue
  // CHECK-NEXT:  |       |   |-ImplicitCastExpr {{.*}} 'const P2718R0::LockGuard &(*)(const P2718R0::LockGuard &)' <FunctionToPointerDecay>
  // CHECK-NEXT:  |       |   | `-DeclRefExpr {{.*}} 'const P2718R0::LockGuard &(const P2718R0::LockGuard &)' lvalue Function {{.*}} 'df1' 'const P2718R0::LockGuard &(const P2718R0::LockGuard &)' (FunctionTemplate {{.*}} 'df1')
  // CHECK-NEXT:  |       |   `-MaterializeTemporaryExpr {{.*}} 'const LockGuard':'const P2718R0::LockGuard' lvalue extended by Var {{.*}} '__range1' 'int (&)[3]'
  // CHECK-NEXT:  |       |     `-ImplicitCastExpr {{.*}} 'const LockGuard':'const P2718R0::LockGuard' <NoOp>
  // CHECK-NEXT:  |       |       `-CXXBindTemporaryExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' (CXXTemporary {{.*}})
  // CHECK-NEXT:  |       |         `-CXXTemporaryObjectExpr {{.*}} 'LockGuard':'P2718R0::LockGuard' 'void ()'
  // CHECK-NEXT:  |       `-DeclRefExpr {{.*}} 'int[3]' lvalue Var {{.*}} 'v' 'int[3]'
  for ([[maybe_unused]] int x : df1(LockGuard()), df1(LockGuard()), v)
    LockGuard guard;
}

// Test default argument && dependent context
template <typename T> int (&default_arg_fn2(const T & = T()))[3];
void test11() {
  for (auto e : default_arg_fn2<A>()) 
    bar(e);
}

template <typename T> A foo2(const T&, const DefaultA &Default = DefaultA());

void test12() {
  for (auto e : default_arg_fn2(foo2(foo2(foo2(A())))))
    bar(e);
}

// Test member call && dependent context
void test13() {

  // CHECK: FunctionDecl {{.*}} test13 'void ()'
  // CHECK:      -CXXForRangeStmt {{.*}}
  // CHECK-NEXT:  |-<<<NULL>>>
  // CHECK-NEXT:  |-DeclStmt {{.*}}
  // CHECK-NEXT:  | `-VarDecl {{.*}} implicit used __range1 'A &&' cinit
  // CHECK-NEXT:  |   `-ExprWithCleanups {{.*}} 'A':'P2718R0::A' xvalue
  // CHECK-NEXT:  |     `-MaterializeTemporaryExpr {{.*}} 'A':'P2718R0::A' xvalue extended by Var {{.*}} '__range1' 'A &&'
  // CHECK-NEXT:  |       `-CXXBindTemporaryExpr {{.*}} 'A':'P2718R0::A' (CXXTemporary {{.*}})
  // CHECK-NEXT:  |         `-CXXMemberCallExpr {{.*}} 'A':'P2718R0::A'
  // CHECK-NEXT:  |           `-MemberExpr {{.*}} '<bound member function type>' .g {{.*}}
  // CHECK-NEXT:  |             `-CXXMemberCallExpr {{.*}} 'A':'P2718R0::A' lvalue
  // CHECK-NEXT:  |               `-MemberExpr {{.*}} '<bound member function type>' .r {{.*}}
  // CHECK-NEXT:  |                 `-MaterializeTemporaryExpr {{.*}} 'A':'P2718R0::A' xvalue extended by Var {{.*}} '__range1' 'A &&'
  // CHECK-NEXT:  |                   `-CXXBindTemporaryExpr {{.*}} 'A':'P2718R0::A' (CXXTemporary {{.*}})
  // CHECK-NEXT:  |                     `-CXXMemberCallExpr {{.*}} 'A':'P2718R0::A'
  // CHECK-NEXT:  |                       `-MemberExpr {{.*}} '<bound member function type>' .g {{.*}}
  // CHECK-NEXT:  |                         `-CXXMemberCallExpr {{.*}} 'A':'P2718R0::A' lvalue
  // CHECK-NEXT:  |                           `-MemberExpr {{.*}} '<bound member function type>' .r {{.*}}
  // CHECK-NEXT:  |                             `-MaterializeTemporaryExpr {{.*}} 'A':'P2718R0::A' xvalue extended by Var {{.*}} '__range1' 'A &&'
  // CHECK-NEXT:  |                               `-CXXBindTemporaryExpr {{.*}} 'A':'P2718R0::A' (CXXTemporary {{.*}})
  // CHECK-NEXT:  |                                 `-CXXMemberCallExpr {{.*}} 'A':'P2718R0::A'
  // CHECK-NEXT:  |                                   `-MemberExpr {{.*}} '<bound member function type>' .g {{.*}}
  // CHECK-NEXT:  |                                     `-CXXMemberCallExpr {{.*}} 'A':'P2718R0::A' lvalue
  // CHECK-NEXT:  |                                       `-MemberExpr {{.*}} '<bound member function type>' .r {{.*}}
  // CHECK-NEXT:  |                                         `-MaterializeTemporaryExpr {{.*}} 'P2718R0::A' xvalue extended by Var {{.*}} '__range1' 'A &&'
  // CHECK-NEXT:  |                                           `-CXXBindTemporaryExpr {{.*}} 'P2718R0::A' (CXXTemporary {{.*}})
  // CHECK-NEXT:  |                                             `-CallExpr {{.*}} 'P2718R0::A'
  // CHECK-NEXT:  |                                               `-ImplicitCastExpr {{.*}} 'P2718R0::A (*)()' <FunctionToPointerDecay>
  // CHECK-NEXT:  |                                                 `-DeclRefExpr {{.*}} 'P2718R0::A ()' lvalue Function {{.*}} 'dg' 'P2718R0::A ()' (FunctionTemplate {{.*}} 'dg')
  for (auto e : dg<A>().r().g().r().g().r().g())
    bar(e);
}
} // namespace P2718R0
