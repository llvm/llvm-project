// RUN: not %clang_cc1 -triple x86_64-unknown-unknown -Wno-unused-value -fcxx-exceptions -std=gnu++17 -ast-dump -frecovery-ast %s | FileCheck -strict-whitespace %s

// Check errors flag is set for RecoveryExpr.
//
// CHECK:     VarDecl {{.*}} a
// CHECK-NEXT:`-RecoveryExpr {{.*}} contains-errors
// CHECK-NEXT:  `-UnresolvedLookupExpr {{.*}} 'bar'
int a = bar();

// The flag propagates through more complicated calls.
//
// CHECK:     VarDecl {{.*}} b
// CHECK-NEXT:`-CallExpr {{.*}} contains-errors
// CHECK-NEXT:  |-UnresolvedLookupExpr {{.*}} 'bar'
// CHECK-NEXT:  |-RecoveryExpr {{.*}} contains-errors
// CHECK-NEXT:  | `-UnresolvedLookupExpr {{.*}} 'baz'
// CHECK-NEXT:   `-RecoveryExpr {{.*}} contains-errors
// CHECK-NEXT:     `-UnresolvedLookupExpr {{.*}} 'qux'
int b = bar(baz(), qux());

// Also propagates through more complicated expressions.
//
// CHECK:     |-VarDecl {{.*}} c
// CHECK-NEXT:| `-BinaryOperator {{.*}} '<dependent type>' contains-errors '*'
// CHECK-NEXT:|   |-UnaryOperator {{.*}} '<dependent type>' contains-errors prefix '&'
// CHECK-NEXT:|   | `-ParenExpr {{.*}} '<dependent type>' contains-errors
// CHECK-NEXT:|   |   `-BinaryOperator {{.*}} '<dependent type>' contains-errors '+'
// CHECK-NEXT:|   |     |-RecoveryExpr {{.*}} '<dependent type>' contains-errors
// CHECK-NEXT:|   |     | `-UnresolvedLookupExpr {{.*}} 'bar'
// CHECK-NEXT:|   |     `-RecoveryExpr {{.*}} '<dependent type>' contains-errors
// CHECK-NEXT:|   |       `-UnresolvedLookupExpr {{.*}} 'baz'
int c = &(bar() + baz()) * 10;

// Errors flag propagates even when type is not dependent anymore.
// CHECK:     |-VarDecl {{.*}} d
// CHECK-NEXT:| `-CXXStaticCastExpr {{.*}} 'int' contains-errors
// CHECK-NEXT:|   `-BinaryOperator {{.*}} '<dependent type>' contains-errors '+'
// CHECK-NEXT:|     |-RecoveryExpr {{.*}} '<dependent type>' contains-errors
// CHECK-NEXT:|     | `-UnresolvedLookupExpr {{.*}} 'bar'
// CHECK-NEXT:|     `-IntegerLiteral {{.*}} 1
int d = static_cast<int>(bar() + 1);


// Error type should result in an invalid decl.
// CHECK: -VarDecl {{.*}} invalid f 'decltype(<recovery-expr>(bar))'
decltype(bar()) f;
