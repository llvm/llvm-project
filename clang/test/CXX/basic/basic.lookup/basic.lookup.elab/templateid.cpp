// RUN: %clang_cc1 -fsyntax-only -verify %s

// elaborated-type-specifier:
//   class-key '::'? nested-name-specifier? 'template'? simple-template-id
// Tests that this form is accepted by the compiler but does not follow
// the elaborated lookup rules of [basic.lookup.elab].

template <typename> class Ident {}; // expected-note {{previous use is here}}

namespace A {
  template <typename> void Ident();

  class Ident<int> AIdent; // expected-error {{refers to a function template}}
  class ::Ident<int> AnotherIdent;
}

class Ident<int> GlobalIdent;
union Ident<int> GlobalIdent2; // expected-error {{ tag type that does not match }}
