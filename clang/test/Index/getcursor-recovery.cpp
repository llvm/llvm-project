int foo(int, int);
int foo(int, double);
int x;

void testTypedRecoveryExpr1() {
  // Inner bar() is an unresolved overloaded call, outer foo() is an overloaded call.
  foo(x, bar(x));
}
// RUN: c-index-test -cursor-at=%s:7:3 %s -Xclang -frecovery-ast -Xclang -frecovery-ast-type | FileCheck -check-prefix=OUTER-FOO %s
// OUTER-FOO: OverloadedDeclRef=foo[2:5, 1:5]
// RUN: c-index-test -cursor-at=%s:7:7 %s -Xclang -frecovery-ast -Xclang -frecovery-ast-type | FileCheck -check-prefix=OUTER-X %s
// OUTER-X: DeclRefExpr=x:3:5
// RUN: c-index-test -cursor-at=%s:7:10 %s -Xclang -frecovery-ast -Xclang -frecovery-ast-type | FileCheck -check-prefix=INNER-FOO %s
// INNER-FOO: OverloadedDeclRef=bar
// RUN: c-index-test -cursor-at=%s:7:14 %s -Xclang -frecovery-ast -Xclang -frecovery-ast-type | FileCheck -check-prefix=INNER-X %s
// INNER-X: DeclRefExpr=x:3:5

void testTypedRecoveryExpr2() {
  // Inner foo() is a RecoveryExpr (with int type), outer foo() is a valid "foo(int, int)" call.
  foo(x, foo(x));
}
// RUN: c-index-test -cursor-at=%s:20:3 %s -Xclang -frecovery-ast -Xclang -frecovery-ast-type | FileCheck -check-prefix=TEST2-OUTER %s
// TEST2-OUTER: DeclRefExpr=foo:1:5
// RUN: c-index-test -cursor-at=%s:20:10 %s -Xclang -frecovery-ast -Xclang -frecovery-ast-type | FileCheck -check-prefix=TEST2-INNER %s
// TEST2-INNER: OverloadedDeclRef=foo[2:5, 1:5]
