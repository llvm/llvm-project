// RUN: %clang_cc1 -std=c++20 -verify %s

template <class T>
  requires(sizeof(T) > 2) || T::value // #FOO_REQ
void Foo(T){};                        // #FOO

template <class T>
void TrailingReturn(T)       // #TRAILING
  requires(sizeof(T) > 2) || // #TRAILING_REQ
          T::value           // #TRAILING_REQ_VAL
{};
template <bool B>
struct HasValue {
  static constexpr bool value = B;
};
static_assert(sizeof(HasValue<true>) <= 2);

template <bool B>
struct HasValueLarge {
  static constexpr bool value = B;
  int I;
};
static_assert(sizeof(HasValueLarge<true>) > 2);

void usage() {
  // Passes the 1st check, short-circuit so the 2nd ::value is not evaluated.
  Foo(1.0);
  TrailingReturn(1.0);

  // Fails the 1st check, but has a ::value, so the check happens correctly.
  Foo(HasValue<true>{});
  TrailingReturn(HasValue<true>{});

  // Passes the 1st check, but would have passed the 2nd one.
  Foo(HasValueLarge<true>{});
  TrailingReturn(HasValueLarge<true>{});

  // Fails the 1st check, fails 2nd because there is no ::value.
  Foo(true);
  // expected-error@-1{{no matching function for call to 'Foo'}}
  // expected-note@#FOO{{candidate template ignored: constraints not satisfied [with T = bool]}}
  // expected-note@#FOO_REQ{{because 'sizeof(_Bool) > 2' (1 > 2) evaluated to false}}
  // expected-note@#FOO_REQ{{because substituted constraint expression is ill-formed: type 'bool' cannot be used prior to '::' because it has no members}}

  TrailingReturn(true);
  // expected-error@-1{{no matching function for call to 'TrailingReturn'}}
  // expected-note@#TRAILING{{candidate template ignored: constraints not satisfied [with T = bool]}}
  // expected-note@#TRAILING_REQ{{because 'sizeof(_Bool) > 2' (1 > 2) evaluated to false}}
  // expected-note@#TRAILING_REQ_VAL{{because substituted constraint expression is ill-formed: type 'bool' cannot be used prior to '::' because it has no members}}

  // Fails the 1st check, fails 2nd because ::value is false.
  Foo(HasValue<false>{});
  // expected-error@-1 {{no matching function for call to 'Foo'}}
  // expected-note@#FOO{{candidate template ignored: constraints not satisfied [with T = HasValue<false>]}}
  // expected-note@#FOO_REQ{{because 'sizeof(HasValue<false>) > 2' (1 > 2) evaluated to false}}
  // expected-note@#FOO_REQ{{and 'HasValue<false>::value' evaluated to false}}
  TrailingReturn(HasValue<false>{});
  // expected-error@-1 {{no matching function for call to 'TrailingReturn'}}
  // expected-note@#TRAILING{{candidate template ignored: constraints not satisfied [with T = HasValue<false>]}}
  // expected-note@#TRAILING_REQ{{because 'sizeof(HasValue<false>) > 2' (1 > 2) evaluated to false}}
  // expected-note@#TRAILING_REQ_VAL{{and 'HasValue<false>::value' evaluated to false}}
}
