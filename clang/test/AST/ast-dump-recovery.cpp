// RUN: not %clang_cc1 -triple x86_64-unknown-unknown -Wno-unused-value -fcxx-exceptions -std=gnu++17 -frecovery-ast -frecovery-ast-type -ast-dump %s | FileCheck -strict-whitespace %s
// RUN: not %clang_cc1 -triple x86_64-unknown-unknown -Wno-unused-value -fcxx-exceptions -std=gnu++17 -fno-recovery-ast -ast-dump %s | FileCheck --check-prefix=DISABLED -strict-whitespace %s

int some_func(int *);

// CHECK:     VarDecl {{.*}} invalid_call
// CHECK-NEXT:  `-RecoveryExpr {{.*}} 'int' contains-errors
// CHECK-NEXT:    |-UnresolvedLookupExpr {{.*}} 'some_func'
// CHECK-NEXT:    `-IntegerLiteral {{.*}} 123
// DISABLED-NOT: -RecoveryExpr {{.*}} contains-errors
int invalid_call = some_func(123);

int some_func2(int a, int b);
void test_invalid_call_2() {
  // CHECK:   -RecoveryExpr {{.*}} '<dependent type>' contains-errors
  // CHECK-NEXT: `-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'some_func2'
  some_func2(,);

  // CHECK:   -RecoveryExpr {{.*}} '<dependent type>' contains-errors
  // CHECK-NEXT: `-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'some_func2'
  some_func2(,,);

  // CHECK:   -RecoveryExpr {{.*}} '<dependent type>' contains-errors
  // CHECK-NEXT: |-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'some_func2'
  // CHECK-NEXT: `-IntegerLiteral {{.*}} 'int' 1
  some_func2(1,);

  // CHECK:   -RecoveryExpr {{.*}} '<dependent type>' contains-errors
  // CHECK-NEXT: |-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'some_func2'
  // CHECK-NEXT: `-IntegerLiteral {{.*}} 'int' 1
  some_func2(,1);
}

int ambig_func(double);
int ambig_func(float);

// CHECK:     VarDecl {{.*}} ambig_call
// CHECK-NEXT:  `-RecoveryExpr {{.*}} 'int' contains-errors
// CHECK-NEXT:    |-UnresolvedLookupExpr {{.*}} 'ambig_func'
// CHECK-NEXT:    `-IntegerLiteral {{.*}} 123
// DISABLED-NOT: -RecoveryExpr {{.*}} contains-errors
int ambig_call = ambig_func(123);

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
  // CHECK:      RecoveryExpr {{.*}} 'ForwardClass':'Foo2::ForwardClass'
  // CHECK-NEXT: `-MemberExpr {{.*}} '<bound member function type>' .createFwd
  // CHECK-NEXT:   `-DeclRefExpr {{.*}} 'f'
  f.createFwd();
  // CHECK:      RecoveryExpr {{.*}} 'int' contains-errors
  // CHECK-NEXT: |-UnresolvedMemberExpr
  // CHECK-NEXT:    `-DeclRefExpr {{.*}} 'Foo2'
  // CHECK-NEXT: `-IntegerLiteral {{.*}} 'int' 1
  f.overload(1);
}

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

  // CHECK:     RecoveryExpr {{.*}} 'Bar' contains-errors
  // CHECK-NEXT:  `-IntegerLiteral {{.*}} 'int' 1
  Bar(1);
}

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

void CtorInitializer() {
  struct S{int m};
  class BaseInit : S {
    BaseInit(float) : S("no match") {}
    // CHECK:      CXXConstructorDecl {{.*}} BaseInit 'void (float)'
    // CHECK-NEXT: |-ParmVarDecl
    // CHECK-NEXT: |-CXXCtorInitializer 'S'
    // CHECK-NEXT: | `-RecoveryExpr {{.*}} 'S'
    // CHECK-NEXT: |   `-StringLiteral
  };
  class DelegatingInit {
    DelegatingInit(float) : DelegatingInit("no match") {}
    // CHECK:      CXXConstructorDecl {{.*}} DelegatingInit 'void (float)'
    // CHECK-NEXT: |-ParmVarDecl
    // CHECK-NEXT: |-CXXCtorInitializer 'DelegatingInit'
    // CHECK-NEXT: | `-RecoveryExpr {{.*}} 'DelegatingInit'
    // CHECK-NEXT: |   `-StringLiteral
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

// Fix crash issue https://github.com/llvm/llvm-project/issues/112560.
// Make sure clang compiles the following code without crashing:

// CHECK:NamespaceDecl {{.*}} GH112560
// CHECK-NEXT:  |-CXXRecordDecl {{.*}} referenced union U definition
// CHECK-NEXT:  | |-DefinitionData {{.*}}
// CHECK-NEXT:  | | |-DefaultConstructor {{.*}}
// CHECK-NEXT:  | | |-CopyConstructor {{.*}}
// CHECK-NEXT:  | | |-MoveConstructor {{.*}}
// CHECK-NEXT:  | | |-CopyAssignment {{.*}}
// CHECK-NEXT:  | | |-MoveAssignment {{.*}}
// CHECK-NEXT:  | | `-Destructor {{.*}}
// CHECK-NEXT:  | |-CXXRecordDecl {{.*}} implicit union U
// CHECK-NEXT:  | `-FieldDecl {{.*}} invalid f 'int'
// CHECK-NEXT:  |   `-RecoveryExpr {{.*}} 'int' contains-errors
// DISABLED-NOT: -RecoveryExpr {{.*}} contains-errors
namespace GH112560 {
union U {
  int f = ;
};

// CHECK: FunctionDecl {{.*}} foo 'void ()'
// CHECK-NEXT:    `-CompoundStmt {{.*}}
// CHECK-NEXT:      `-DeclStmt {{.*}}
// CHECK-NEXT:        `-VarDecl {{.*}} g 'U':'GH112560::U' listinit
// CHECK-NEXT:          `-InitListExpr {{.*}} 'U':'GH112560::U' contains-errors field Field {{.*}} 'f' 'int'
// CHECK-NEXT:            `-CXXDefaultInitExpr {{.*}} 'int' contains-errors has rewritten init
// CHECK-NEXT:              `-RecoveryExpr {{.*}} 'int' contains-errors
// DISABLED-NOT: -RecoveryExpr {{.*}} contains-errors
void foo() {
  U g{};
}
} // namespace GH112560
