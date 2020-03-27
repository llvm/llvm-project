// RUN: not %clang_cc1 -triple x86_64-unknown-unknown -Wno-unused-value -fcxx-exceptions -std=gnu++17 -frecovery-ast -ast-dump %s | FileCheck -strict-whitespace %s
// RUN: not %clang_cc1 -triple x86_64-unknown-unknown -Wno-unused-value -fcxx-exceptions -std=gnu++17 -fno-recovery-ast -ast-dump %s | FileCheck --check-prefix=DISABLED -strict-whitespace %s

int some_func(int *);

// CHECK:     VarDecl {{.*}} invalid_call
// CHECK-NEXT:`-RecoveryExpr {{.*}} contains-errors
// CHECK-NEXT:  |-UnresolvedLookupExpr {{.*}} 'some_func'
// CHECK-NEXT:  `-IntegerLiteral {{.*}} 123
// DISABLED-NOT: -RecoveryExpr {{.*}} contains-errors
int invalid_call = some_func(123);

int ambig_func(double);
int ambig_func(float);

// CHECK:     VarDecl {{.*}} ambig_call
// CHECK-NEXT:`-RecoveryExpr {{.*}} contains-errors
// CHECK-NEXT:  |-UnresolvedLookupExpr {{.*}} 'ambig_func'
// CHECK-NEXT:  `-IntegerLiteral {{.*}} 123
// DISABLED-NOT: -RecoveryExpr {{.*}} contains-errors
int ambig_call = ambig_func(123);

// CHECK:     VarDecl {{.*}} unresolved_call1
// CHECK-NEXT:`-RecoveryExpr {{.*}} contains-errors
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