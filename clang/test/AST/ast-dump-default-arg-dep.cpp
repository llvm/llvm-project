// RUN: not %clang_cc1 -triple x86_64-unknown-unknown -ast-dump -frecovery-ast %s | FileCheck %s

// CXXDefaultArgExpr should inherit dependence from the inner Expr, in this case
// RecoveryExpr.
void fun(int arg = foo());

void test() {
  fun();
}
// CHECK: -CXXDefaultArgExpr 0x{{[^ ]*}} <<invalid sloc>> '<dependent type>' contains-errors lvalue
