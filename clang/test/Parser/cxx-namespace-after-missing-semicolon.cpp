// RUN: %clang_cc1 -fsyntax-only -verify %s

UNKNOWN_MACRO_1("z", 1) // expected-error {{a type specifier is required for all declarations}}
// expected-error@-1 {{expected ';' after top level declarator}}

namespace foo {
  class bar {};
}

int variable = 0; // ok
foo::bar something; // ok

UNKNOWN_MACRO_2(void) // expected-error {{a type specifier is required for all declarations}}
// expected-error@-1 {{expected ';' after top level declarator}}

namespace baz {
  using Bool = bool;
}

int variable2 = 2; // ok
const baz::Bool flag = false;  // ok
