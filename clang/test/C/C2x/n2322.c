// RUN: %clang_cc1 -std=c2x -ast-dump %s | FileCheck %s

/* WG14 N2322: partial
 * Preprocessor line numbers unspecified
 */
void n2322() {
  // The line number associated with a pp-token should be the line number of
  // the first character of the pp-token.
  "this string literal  \
   spans multiple lines \
   before terminating";
// CHECK: ImplicitCastExpr {{.*}} <line:9
// CHECK-NEXT: StringLiteral {{.*}} <col:3>

  // The line number associated with a pp-directive should be the line number
  // of the line with the first # token.
  // Possible FIXME: The AST node should be on line 1002 if we take the line
  // number to be associated with the first # token. However, this relies on an
  // interpretation of the standard definition of "presumed line" to be before
  // line splices are removed. The standard leaves this unspecified, so this
  // may not represent an actual issue.
  #\
  line\
  1000
  "string literal";
// CHECK: ImplicitCastExpr {{.*}} <line:1000
// CHECK: StringLiteral {{.*}} <col:3>

  // The line number associated with a macro invocation should be the line
  // number of the first character of the macro name in the invocation.
  //
  // Reset the line number to make it easier to understand the next test.
  // FIXME: The line number should be 2005 (first letter of the macro name) and
  // not 2007 (closing parenthesis of the macro invocation).
  #line 2000
  #define F( \
    )        \
_\
_LINE__

  _Static_assert(F(\
  \
  ) == 2007);

  // Reset the line number again for ease.
  #line 2000
  _Static_assert(2001 == \
__LI\
NE__\
  );
}
