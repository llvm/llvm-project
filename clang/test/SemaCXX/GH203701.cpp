// RUN: %clang_cc1 -std=c++2b -fsyntax-only -verify %s

namespace GH203701 {
  struct S {
    constexpr S(auto) {}
    constexpr operator int() const { return 0; }
  };

  constexpr auto a = [](this S) { return 1; };

  static_assert((&decltype(a)::operator())(1) == 42, ""); // expected-error-re {{static assertion failed due to requirement '(&const GH203701::(lambda at {{.*}})::operator())(1) == 42'{{.*}}}} \
                                                           // expected-note {{expression evaluates to '1 == 42'}}
  static_assert((&S::operator int) == nullptr, ""); // expected-error {{static assertion failed due to requirement '(&GH203701::S::operator int) == nullptr'}}
}
