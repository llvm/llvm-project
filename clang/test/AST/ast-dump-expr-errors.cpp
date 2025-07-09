// RUN: not %clang_cc1 -triple x86_64-unknown-unknown -Wno-unused-value -fcxx-exceptions -std=gnu++17 -ast-dump -frecovery-ast %s | FileCheck -strict-whitespace %s

int a = bar();
// CHECK: |-VarDecl {{.*}} a 'int' cinit
// CHECK: | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | `-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'bar' empty
// CHECK: | `-typeDetails: BuiltinType {{.*}} 'int'

// The flag propagates through more complicated calls.
int b = bar(baz(), qux());
// CHECK: |-VarDecl {{.*}} b 'int' cinit
// CHECK: | |-CallExpr {{.*}} '<dependent type>' contains-errors
// CHECK: | | |-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'bar' empty
// CHECK: | | |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | | `-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'baz' empty
// CHECK: | | `-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | |   `-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'qux' empty
// CHECK: | `-typeDetails: BuiltinType {{.*}} 'int'

// Also propagates through more complicated expressions.
int c = &(bar() + baz()) * 10;

// CHECK: |-VarDecl {{.*}} c 'int' cinit
// CHECK: | |-BinaryOperator {{.*}} '<dependent type>' contains-errors '*'
// CHECK: | | |-UnaryOperator {{.*}} '<dependent type>' contains-errors prefix '&' cannot overflow
// CHECK: | | | `-ParenExpr {{.*}} '<dependent type>' contains-errors
// CHECK: | | |   `-BinaryOperator {{.*}} '<dependent type>' contains-errors '+'
// CHECK: | | |     |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | | |     | `-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'bar' empty
// CHECK: | | |     `-RecoveryExpr {{.*}}'<dependent type>' contains-errors lvalue
// CHECK: | | |       `-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'baz' empty
// CHECK: | | `-IntegerLiteral {{.*}} 'int' 10
// CHECK: | `-typeDetails: BuiltinType {{.*}} 'int'

// Errors flag propagates even when type is not dependent anymore.
int d = static_cast<int>(bar() + 1);
// CHECK: |-VarDecl {{.*}} d 'int' cinit
// CHECK: | |-CXXStaticCastExpr {{.*}} 'int' contains-errors static_cast<int> <Dependent>
// CHECK: | | `-BinaryOperator {{.*}} '<dependent type>' contains-errors '+'
// CHECK: | |   |-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK: | |   | `-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'bar' empty
// CHECK: | |   `-IntegerLiteral {{.*}} 'int' 1
// CHECK: | `-typeDetails: BuiltinType {{.*}} 'int'


// Error type should result in an invalid decl.

decltype(bar()) f;
// CHECK: `-VarDecl {{.*}} invalid f 'decltype(<recovery-expr>(bar))'
// CHECK:   `-typeDetails: DecltypeType {{.*}} 'decltype(<recovery-expr>(bar))' contains-errors dependent
// CHECK:     `-RecoveryExpr {{.*}} '<dependent type>' contains-errors lvalue
// CHECK:       `-UnresolvedLookupExpr {{.*}} '<overloaded function type>' lvalue (ADL) = 'bar' empty