// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

constexpr void gh195416() {
  struct U {
    struct S {};
    static constexpr S S::bar;
    // expected-error@-1 {{non-friend class member 'bar' cannot have a qualified name}}
    // expected-error@-2 {{static data member 'bar' not allowed in local struct 'S'}}
  };
}
