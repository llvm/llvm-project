// RUN: not %clang_cc1 -triple x86_64-unknown-unknown -Wno-unused-value -fcxx-exceptions -std=gnu++17 -frecovery-ast -frecovery-ast-type -ast-dump %s | FileCheck -strict-whitespace %s
// RUN: not %clang_cc1 -triple x86_64-unknown-unknown -Wno-unused-value -fcxx-exceptions -std=gnu++17 -fno-recovery-ast -ast-dump %s | FileCheck --check-prefix=DISABLED -strict-whitespace %s

int some_func(int *);

// CHECK:     VarDecl {{.*}} invalid_call
// CHECK-NEXT:  `-RecoveryExpr {{.*}} 'int' contains-errors
// CHECK-NEXT:    |-UnresolvedLookupExpr {{.*}} 'some_func'
// CHECK-NEXT:    `-IntegerLiteral {{.*}} 123
// DISABLED-NOT: -RecoveryExpr {{.*}} contains-errors
int invalid_call = some_func(123);
void test_invalid_call(int s) {
  // CHECK:      CallExpr {{.*}} '<dependent type>' contains-errors
  // CHECK-NEXT: |-UnresolvedLookupExpr {{.*}} 'some_func'
  // CHECK-NEXT: |-RecoveryExpr {{.*}} <col:13>
  // CHECK-NEXT: `-BinaryOperator {{.*}}
  // CHECK-NEXT:   |-RecoveryExpr {{.*}}
  // CHECK-NEXT:   `-IntegerLiteral {{.*}} <col:28> 'int' 1
  some_func(undef1, undef2+1);

  // CHECK:      BinaryOperator {{.*}} '<dependent type>' contains-errors '='
  // CHECK-NEXT: |-DeclRefExpr {{.*}} 's'
  // CHECK-NEXT: `-CallExpr {{.*}} '<dependent type>' contains-errors
  // CHECK-NEXT:   |-UnresolvedLookupExpr {{.*}} 'some_func'
  // CHECK-NEXT:   `-RecoveryExpr {{.*}} contains-errors
  s = some_func(undef1);

  // CHECK:     VarDecl {{.*}} var 'int'
  // CHECK-NEXT: `-CallExpr {{.*}} '<dependent type>' contains-errors
  // CHECK-NEXT:   |-UnresolvedLookupExpr {{.*}} 'some_func'
  // CHECK-NEXT:   `-RecoveryExpr {{.*}} contains-errors
  int var = some_func(undef1);
}

int ambig_func(double);
int ambig_func(float);

// CHECK:     VarDecl {{.*}} ambig_call
// CHECK-NEXT:  `-RecoveryExpr {{.*}} 'int' contains-errors
// CHECK-NEXT:    |-UnresolvedLookupExpr {{.*}} 'ambig_func'
// CHECK-NEXT:    `-IntegerLiteral {{.*}} 123
// DISABLED-NOT: -RecoveryExpr {{.*}} contains-errors
int ambig_call = ambig_func(123);

// CHECK:     VarDecl {{.*}} unresolved_call1
// CHECK-NEXT:`-RecoveryExpr {{.*}} '<dependent type>' contains-errors
// CHECK-NEXT:  `-UnresolvedLookupExpr {{.*}} 'bar'
// DISABLED-NOT: -RecoveryExpr {{.*}} contains-errors
int unresolved_call1 = bar();

// CHECK:     VarDecl {{.*}} unresolved_call2
// CHECK-NEXT:`-CallExpr {{.*}} contains-errors
// CHECK-NEXT:  |-UnresolvedLookupExpr {{.*}} 'bar'
// CHECK-NEXT:  |-RecoveryExpr {{.*}} contains-errors
// CHECK-NEXT:  | `-UnresolvedLookupExpr {{.*}} 'baz'
// CHECK-NEXT:   `-RecoveryExpr {{.*}} contains-errors
// CHECK-NEXT:     `-UnresolvedLookupExpr {{.*}} 'qux'
// DISABLED-NOT: -RecoveryExpr {{.*}} contains-errors
int unresolved_call2 = bar(baz(), qux());

constexpr int a = 10;

// CHECK:     VarDecl {{.*}} postfix_inc
// CHECK-NEXT:`-RecoveryExpr {{.*}} contains-errors
// CHECK-NEXT:  `-DeclRefExpr {{.*}} 'a'
// DISABLED-NOT: -RecoveryExpr {{.*}} contains-errors
int postfix_inc = a++;

// CHECK:     VarDecl {{.*}} prefix_inc
// CHECK-NEXT:`-RecoveryExpr {{.*}} contains-errors
// CHECK-NEXT:  `-DeclRefExpr {{.*}} 'a'
// DISABLED-NOT: -RecoveryExpr {{.*}} contains-errors
int prefix_inc = ++a;

// CHECK:     VarDecl {{.*}} unary_address
// CHECK-NEXT:`-RecoveryExpr {{.*}} contains-errors
// CHECK-NEXT:  `-ParenExpr {{.*}}
// CHECK-NEXT:    `-BinaryOperator {{.*}} '+'
// CHECK-NEXT:      |-ImplicitCastExpr
// CHECK-NEXT:      | `-DeclRefExpr {{.*}} 'a'
// DISABLED-NOT: -RecoveryExpr {{.*}} contains-errors
int unary_address = &(a + 1);

// CHECK:     VarDecl {{.*}} unary_bitinverse
// CHECK-NEXT:`-RecoveryExpr {{.*}} contains-errors
// CHECK-NEXT:  `-ParenExpr {{.*}}
// CHECK-NEXT:    `-BinaryOperator {{.*}} '+'
// CHECK-NEXT:      |-ImplicitCastExpr
// CHECK-NEXT:      | `-ImplicitCastExpr
// CHECK-NEXT:      |   `-DeclRefExpr {{.*}} 'a'
// DISABLED-NOT: -RecoveryExpr {{.*}} contains-errors
int unary_bitinverse = ~(a + 0.0);

// CHECK:     VarDecl {{.*}} binary
// CHECK-NEXT:`-RecoveryExpr {{.*}} contains-errors
// CHECK-NEXT:  |-DeclRefExpr {{.*}} 'a'
// CHECK-NEXT:  `-CXXNullPtrLiteralExpr
// DISABLED-NOT: -RecoveryExpr {{.*}} contains-errors
int binary = a + nullptr;

// CHECK:     VarDecl {{.*}} ternary
// CHECK-NEXT:`-RecoveryExpr {{.*}} contains-errors
// CHECK-NEXT:  |-DeclRefExpr {{.*}} 'a'
// CHECK-NEXT:  |-CXXNullPtrLiteralExpr
// CHECK-NEXT:  `-DeclRefExpr {{.*}} 'a'
// DISABLED-NOT: -RecoveryExpr {{.*}} contains-errors
int ternary = a ? nullptr : a;

// CHECK:     FunctionDecl
// CHECK-NEXT:|-ParmVarDecl {{.*}} x
// CHECK-NEXT:`-CompoundStmt
// CHECK-NEXT: |-RecoveryExpr {{.*}} contains-errors
// CHECK-NEXT: | `-DeclRefExpr {{.*}} 'foo'
// CHECK-NEXT: `-CallExpr {{.*}} contains-errors
// CHECK-NEXT:  |-RecoveryExpr {{.*}} contains-errors
// CHECK-NEXT:  | `-DeclRefExpr {{.*}} 'foo'
// CHECK-NEXT:  `-DeclRefExpr {{.*}} 'x'
struct Foo {} foo;
void test(int x) {
  foo.abc;
  foo->func(x);
}

void AccessIncompleteClass() {
  struct Forward;
  Forward* ptr;
  // CHECK:      CallExpr {{.*}} '<dependent type>'
  // CHECK-NEXT: `-CXXDependentScopeMemberExpr {{.*}} '<dependent type>'
  // CHECK-NEXT:   `-RecoveryExpr {{.*}} '<dependent type>' contains-errors
  // CHECK-NEXT:     `-DeclRefExpr {{.*}} 'Forward *'
  ptr->method();
}

struct Foo2 {
  double func();
  class ForwardClass;
  ForwardClass createFwd();

  int overload();
  int overload(int, int);
};
void test2(Foo2 f) {
  // CHECK:      RecoveryExpr {{.*}} 'double'
  // CHECK-NEXT:   |-MemberExpr {{.*}} '<bound member function type>'
  // CHECK-NEXT:   | `-DeclRefExpr {{.*}} 'f'
  // CHECK-NEXT: `-IntegerLiteral {{.*}} 'int' 1
  f.func(1);
  // CHECK:      RecoveryExpr {{.*}} 'Foo2::ForwardClass'
  // CHECK-NEXT: `-MemberExpr {{.*}} '<bound member function type>' .createFwd
  // CHECK-NEXT:   `-DeclRefExpr {{.*}} 'f'
  f.createFwd();
  // CHECK:      RecoveryExpr {{.*}} 'int' contains-errors
  // CHECK-NEXT: |-UnresolvedMemberExpr
  // CHECK-NEXT:    `-DeclRefExpr {{.*}} 'Foo2'
  // CHECK-NEXT: `-IntegerLiteral {{.*}} 'int' 1
  f.overload(1);
}

// CHECK:     |-AlignedAttr {{.*}} alignas
// CHECK-NEXT:| `-RecoveryExpr {{.*}} contains-errors
// CHECK-NEXT:|   `-UnresolvedLookupExpr {{.*}} 'invalid'
struct alignas(invalid()) Aligned {};

auto f();
int f(double);
// CHECK:      VarDecl {{.*}} unknown_type_call 'int'
// CHECK-NEXT: `-RecoveryExpr {{.*}} '<dependent type>'
int unknown_type_call = f(0, 0);

void InvalidInitalizer(int x) {
  struct Bar { Bar(); };
  // CHECK:     `-VarDecl {{.*}} a1 'Bar'
  // CHECK-NEXT: `-RecoveryExpr {{.*}} contains-errors
  // CHECK-NEXT:  `-IntegerLiteral {{.*}} 'int' 1
  Bar a1(1);
  // CHECK:     `-VarDecl {{.*}} a2 'Bar'
  // CHECK-NEXT: `-RecoveryExpr {{.*}} contains-errors
  // CHECK-NEXT:  `-DeclRefExpr {{.*}} 'x'
  Bar a2(x);
  // CHECK:     `-VarDecl {{.*}} a3 'Bar'
  // CHECK-NEXT: `-RecoveryExpr {{.*}} contains-errors
  // CHECK-NEXT:  `-InitListExpr
  // CHECK-NEDT:   `-DeclRefExpr {{.*}} 'x'
  Bar a3{x};
  // CHECK:     `-VarDecl {{.*}} a4 'Bar'
  // CHECK-NEXT: `-ParenListExpr {{.*}} 'NULL TYPE' contains-errors
  // CHECK-NEXT:  `-RecoveryExpr {{.*}} contains-errors
  // CHECK-NEXT:   `-UnresolvedLookupExpr {{.*}} 'invalid'
  Bar a4(invalid());
  // CHECK:     `-VarDecl {{.*}} a5 'Bar'
  // CHECK-NEXT: `-InitListExpr {{.*}} contains-errors
  // CHECK-NEXT:  `-RecoveryExpr {{.*}} contains-errors
  // CHECK-NEXT:   `-UnresolvedLookupExpr {{.*}} 'invalid'
  Bar a5{invalid()};

  // CHECK:     `-VarDecl {{.*}} b1 'Bar'
  // CHECK-NEXT: `-RecoveryExpr {{.*}} contains-errors
  // CHECK-NEXT:  `-IntegerLiteral {{.*}} 'int' 1
  Bar b1 = 1;
  // CHECK:     `-VarDecl {{.*}} b2 'Bar'
  // CHECK-NEXT: `-RecoveryExpr {{.*}} contains-errors
  // CHECK-NEXT:  `-InitListExpr
  Bar b2 = {1};
  // CHECK:     `-VarDecl {{.*}} b3 'Bar'
  // CHECK-NEXT:  `-RecoveryExpr {{.*}} 'Bar' contains-errors
  // CHECK-NEXT:    `-DeclRefExpr {{.*}} 'x' 'int'
  Bar b3 = Bar(x);
  // CHECK:     `-VarDecl {{.*}} b4 'Bar'
  // CHECK-NEXT:  `-RecoveryExpr {{.*}} 'Bar' contains-errors
  // CHECK-NEXT:    `-InitListExpr {{.*}} 'void'
  // CHECK-NEXT:      `-DeclRefExpr {{.*}} 'x' 'int'
  Bar b4 = Bar{x};
  // CHECK:     `-VarDecl {{.*}} b5 'Bar'
  // CHECK-NEXT: `-CXXUnresolvedConstructExpr {{.*}} 'Bar' contains-errors 'Bar'
  // CHECK-NEXT:   `-RecoveryExpr {{.*}} contains-errors
  // CHECK-NEXT:     `-UnresolvedLookupExpr {{.*}} 'invalid'
  Bar b5 = Bar(invalid());
  // CHECK:     `-VarDecl {{.*}} b6 'Bar'
  // CHECK-NEXT: `-CXXUnresolvedConstructExpr {{.*}} 'Bar' contains-errors 'Bar'
  // CHECK-NEXT:  `-InitListExpr {{.*}} contains-errors
  // CHECK-NEXT:   `-RecoveryExpr {{.*}} contains-errors
  // CHECK-NEXT:     `-UnresolvedLookupExpr {{.*}} 'invalid'
  Bar b6 = Bar{invalid()};

  // CHECK:     RecoveryExpr {{.*}} 'Bar' contains-errors
  // CHECK-NEXT:  `-IntegerLiteral {{.*}} 'int' 1
  Bar(1);

  // CHECK:     `-VarDecl {{.*}} var1
  // CHECK-NEXT: `-BinaryOperator {{.*}} '<dependent type>' contains-errors
  // CHECK-NEXT:   |-RecoveryExpr {{.*}} '<dependent type>' contains-errors
  // CHECK-NEXT:   `-IntegerLiteral {{.*}} 'int' 1
  int var1 = undef + 1;
}
void InitializerForAuto() {
  // CHECK:     `-VarDecl {{.*}} invalid a 'auto'
  // CHECK-NEXT: `-RecoveryExpr {{.*}} '<dependent type>' contains-errors
  // CHECK-NEXT:   `-UnresolvedLookupExpr {{.*}} 'invalid'
  auto a = invalid();

  // CHECK:     `-VarDecl {{.*}} invalid b 'auto'
  // CHECK-NEXT: `-CallExpr {{.*}} '<dependent type>' contains-errors
  // CHECK-NEXT:   |-UnresolvedLookupExpr {{.*}} 'some_func'
  // CHECK-NEXT:   `-RecoveryExpr {{.*}} '<dependent type>' contains-errors
  // CHECK-NEXT:     `-UnresolvedLookupExpr {{.*}} 'invalid'
  auto b = some_func(invalid());

  decltype(ned);
  // very bad initailizer: there is an unresolved typo expr internally, we just
  // drop it.
  // CHECK: `-VarDecl {{.*}} invalid unresolved_typo 'auto'
  auto unresolved_typo = gned.*[] {};
}

// Verified that the generated call operator is invalid.
// CHECK: |-CXXMethodDecl {{.*}} invalid operator() 'auto () const -> auto'
using Escape = decltype([] { return undef(); }());

// CHECK:      VarDecl {{.*}} NoCrashOnInvalidInitList
// CHECK-NEXT: `-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK-NEXT:   `-InitListExpr
// CHECK-NEXT:     `-DesignatedInitExpr {{.*}} 'void'
// CHECK-NEXT:       `-CXXNullPtrLiteralExpr {{.*}} 'std::nullptr_t'
struct {
  int& abc;
} NoCrashOnInvalidInitList = {
  .abc = nullptr,
};

// Verify the value category of recovery expression.
int prvalue(int);
int &lvalue(int);
int &&xvalue(int);
void ValueCategory() {
  // CHECK:  RecoveryExpr {{.*}} 'int' contains-errors
  prvalue(); // call to a function (nonreference return type) yields a prvalue (not print by default)
  // CHECK:  RecoveryExpr {{.*}} 'int' contains-errors lvalue
  lvalue(); // call to a function (lvalue reference return type) yields an lvalue.
  // CHECK:  RecoveryExpr {{.*}} 'int' contains-errors xvalue
  xvalue(); // call to a function (rvalue reference return type) yields an xvalue.
}

void InvalidCondition() {
  // CHECK:      IfStmt {{.*}}
  // CHECK-NEXT: |-RecoveryExpr {{.*}} <col:7, col:15> '<dependent type>' contains-errors
  // CHECK-NEXT: | `-UnresolvedLookupExpr {{.*}} <col:7>
  if (invalid()) {}

  // CHECK:      WhileStmt {{.*}}
  // CHECK-NEXT: |-RecoveryExpr {{.*}} <col:10, col:18> '<dependent type>' contains-errors
  // CHECK-NEXT: | `-UnresolvedLookupExpr {{.*}} <col:10>
  while (invalid()) {}

  // CHECK:      SwitchStmt {{.*}}
  // CHECK-NEXT: |-RecoveryExpr {{.*}} '<dependent type>' contains-errors
  // CHECK-NEXT: | `-UnresolvedLookupExpr {{.*}} <col:10>
  switch(invalid()) {
    case 1:
      break;
  }
  // FIXME: figure out why the type of ConditionalOperator is not int.
  // CHECK:      ConditionalOperator {{.*}} '<dependent type>' contains-errors
  // CHECK-NEXT: |-RecoveryExpr {{.*}} '<dependent type>' contains-errors
  // CHECK-NEXT: | `-UnresolvedLookupExpr {{.*}}
  // CHECK-NEXT: |-IntegerLiteral {{.*}} 'int' 1
  // CHECK-NEXT: `-IntegerLiteral {{.*}} 'int' 2
  invalid() ? 1 : 2;
}

void CtorInitializer() {
  struct S{int m};
  class MemberInit {
    int x, y, z;
    S s;
    MemberInit() : x(invalid), y(invalid, invalid), z(invalid()), s(1,2) {}
    // CHECK:      CXXConstructorDecl {{.*}} MemberInit 'void ()'
    // CHECK-NEXT: |-CXXCtorInitializer Field {{.*}} 'x' 'int'
    // CHECK-NEXT: | `-ParenListExpr
    // CHECK-NEXT: |   `-RecoveryExpr {{.*}} '<dependent type>'
    // CHECK-NEXT: |-CXXCtorInitializer Field {{.*}} 'y' 'int'
    // CHECK-NEXT: | `-ParenListExpr
    // CHECK-NEXT: |   |-RecoveryExpr {{.*}} '<dependent type>'
    // CHECK-NEXT: |   `-RecoveryExpr {{.*}} '<dependent type>'
    // CHECK-NEXT: |-CXXCtorInitializer Field {{.*}} 'z' 'int'
    // CHECK-NEXT: | `-ParenListExpr
    // CHECK-NEXT: |   `-RecoveryExpr {{.*}} '<dependent type>'
    // CHECK-NEXT: |     `-UnresolvedLookupExpr {{.*}} '<overloaded function type>'
    // CHECK-NEXT: |-CXXCtorInitializer Field {{.*}} 's' 'S'
    // CHECK-NEXT: | `-RecoveryExpr {{.*}} 'S' contains-errors
    // CHECK-NEXT: |   |-IntegerLiteral {{.*}} 1
    // CHECK-NEXT: |   `-IntegerLiteral {{.*}} 2
  };
  class BaseInit : S {
    BaseInit(float) : S("no match") {}
    // CHECK:      CXXConstructorDecl {{.*}} BaseInit 'void (float)'
    // CHECK-NEXT: |-ParmVarDecl
    // CHECK-NEXT: |-CXXCtorInitializer 'S'
    // CHECK-NEXT: | `-RecoveryExpr {{.*}} 'S'
    // CHECK-NEXT: |   `-StringLiteral

    BaseInit(double) : S(invalid) {}
    // CHECK:      CXXConstructorDecl {{.*}} BaseInit 'void (double)'
    // CHECK-NEXT: |-ParmVarDecl
    // CHECK-NEXT: |-CXXCtorInitializer 'S'
    // CHECK-NEXT: | `-ParenListExpr
    // CHECK-NEXT: |   `-RecoveryExpr {{.*}} '<dependent type>'
  };
  class DelegatingInit {
    DelegatingInit(float) : DelegatingInit("no match") {}
    // CHECK:      CXXConstructorDecl {{.*}} DelegatingInit 'void (float)'
    // CHECK-NEXT: |-ParmVarDecl
    // CHECK-NEXT: |-CXXCtorInitializer 'DelegatingInit'
    // CHECK-NEXT: | `-RecoveryExpr {{.*}} 'DelegatingInit'
    // CHECK-NEXT: |   `-StringLiteral

    DelegatingInit(double) : DelegatingInit(invalid) {}
    // CHECK:      CXXConstructorDecl {{.*}} DelegatingInit 'void (double)'
    // CHECK-NEXT: |-ParmVarDecl
    // CHECK-NEXT: |-CXXCtorInitializer 'DelegatingInit'
    // CHECK-NEXT: | `-ParenListExpr
    // CHECK-NEXT: |   `-RecoveryExpr {{.*}} '<dependent type>'
  };
}

float *brokenReturn() {
  // CHECK:      FunctionDecl {{.*}} brokenReturn
  return 42;
  // CHECK:      ReturnStmt
  // CHECK-NEXT: `-RecoveryExpr {{.*}} 'float *'
  // CHECK-NEXT:   `-IntegerLiteral {{.*}} 'int' 42
}

// Return deduction treats the first, second *and* third differently!
auto *brokenDeducedReturn(int *x, float *y, double *z) {
  // CHECK:      FunctionDecl {{.*}} invalid brokenDeducedReturn
  if (x) return x;
  // CHECK:      ReturnStmt
  // CHECK-NEXT: `-ImplicitCastExpr {{.*}} <LValueToRValue>
  // CHECK-NEXT:   `-DeclRefExpr {{.*}} 'x' 'int *'
  if (y) return y;
  // CHECK:      ReturnStmt
  // CHECK-NEXT: `-RecoveryExpr {{.*}} 'int *'
  // CHECK-NEXT:   `-DeclRefExpr {{.*}} 'y' 'float *'
  if (z) return z;
  // CHECK:      ReturnStmt
  // CHECK-NEXT: `-RecoveryExpr {{.*}} 'int *'
  // CHECK-NEXT:   `-DeclRefExpr {{.*}} 'z' 'double *'
  return x;
  // Unfortunate: we wrap a valid return in RecoveryExpr.
  // This is to avoid running deduction again after it failed once.
  // CHECK:      ReturnStmt
  // CHECK-NEXT: `-RecoveryExpr {{.*}} 'int *'
  // CHECK-NEXT:   `-DeclRefExpr {{.*}} 'x' 'int *'
}

void returnInitListFromVoid() {
  // CHECK:      FunctionDecl {{.*}} returnInitListFromVoid
  return {7,8};
  // CHECK:      ReturnStmt
  // CHECK-NEXT: `-RecoveryExpr {{.*}} '<dependent type>'
  // CHECK-NEXT:   |-IntegerLiteral {{.*}} 'int' 7
  // CHECK-NEXT:   `-IntegerLiteral {{.*}} 'int' 8
}
