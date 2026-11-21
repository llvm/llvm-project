// RUN: %clang_cc1 -fsyntax-only -verify=expected,c %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,c -std=c2y %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,objc -x objective-c %s

// Outside C++, '::' cannot start a declaration, so the tentative-parse
// disambiguation must report it as such and parse the condition as an
// expression. Recursing on the same token never makes progress, which used to
// overflow the stack in C. In Objective-C it produced a cascade of bogus
// declaration diagnostics instead of a single 'expected a type'.
// https://github.com/llvm/llvm-project/issues/243727

void reduced(void) { if (::0); }
// c-error@-1 {{expected expression}}
// objc-error@-2 {{expected a type}}
// expected-warning@-3 {{if statement has empty body}}
// expected-note@-4 {{put the semicolon on a separate line to silence this warning}}

void if_condition(void) { if (::0) {} }
// c-error@-1 {{expected expression}}
// objc-error@-2 {{expected a type}}

void while_condition(void) { while (::0) {} }
// c-error@-1 {{expected expression}}
// objc-error@-2 {{expected a type}}

void do_while_condition(void) { do {} while (::0); }
// c-error@-1 {{expected expression}}
// objc-error@-2 {{expected a type}}

void switch_condition(void) { switch (::0) {} }
// c-error@-1 {{expected expression}}
// objc-error@-2 {{expected a type}}

void for_init_clause(void) { for (::0;;) {} }
// c-error@-1 {{expected expression}}
// objc-error@-2 {{expected a type}}

void for_condition_clause(void) { for (; ::0;) {} }
// c-error@-1 {{expected expression}}
// objc-error@-2 {{expected a type}}
