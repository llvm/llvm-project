int foo(int, int);
int foo(int, double);
int x;

void testTypedRecoveryExpr() {
  // Inner foo() is a RecoveryExpr, outer foo() is an overloaded call.
  foo(x, foo(x));
}
// RUN: c-index-test -cursor-at=%s:7:3 %s -Xclang -frecovery-ast | FileCheck -check-prefix=OUTER-FOO %s
// OUTER-FOO: OverloadedDeclRef=foo[2:5, 1:5]
// RUN: c-index-test -cursor-at=%s:7:7 %s -Xclang -frecovery-ast | FileCheck -check-prefix=OUTER-X %s
// OUTER-X: DeclRefExpr=x:3:5
// RUN: c-index-test -cursor-at=%s:7:10 %s -Xclang -frecovery-ast | FileCheck -check-prefix=INNER-FOO %s
// INNER-FOO: OverloadedDeclRef=foo[2:5, 1:5]
// RUN: c-index-test -cursor-at=%s:7:14 %s -Xclang -frecovery-ast | FileCheck -check-prefix=INNER-X %s
// INNER-X: DeclRefExpr=x:3:5
