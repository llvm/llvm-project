// RUN: %clang_cc1 -fsyntax-only -pedantic -verify %s

// This syntax error used to cause use-after free due to token local buffer
// in ParseCXXAmbiguousParenExpression.
int H((int()[)]);
// expected-error@-1 {{expected expression}}
// expected-error@-2 {{expected ']'}}
// expected-note@-3 {{to match this '['}}
// expected-error@-4 {{expected ';' after top level declarator}}
