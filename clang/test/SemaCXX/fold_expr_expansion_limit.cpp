// RUN: %clang_cc1 -fsyntax-only -fbracket-depth 2 -verify -std=c++17 %s

template <class T, T... V> struct seq {
  constexpr bool zero() { return (true && ... && (V == 0)); }; // expected-error {{instantiating fold expression with 3 arguments exceeded expression nesting limit of 2}} \
                                                                  expected-note {{use -fbracket-depth}}
};
constexpr unsigned N = 3;
auto x = __make_integer_seq<seq, int, N>{};
static_assert(!x.zero(), ""); // expected-note {{in instantiation of member function}}
