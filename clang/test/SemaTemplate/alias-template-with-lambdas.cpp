// RUN: %clang_cc1 -std=c++2c -fsyntax-only -verify %s
namespace lambda_calls {

template <class>
concept True = true;

template <class>
concept False = false; // #False

template <class T> struct S {
  template <class... U> using type = decltype([](U...) {}(U()...));
  template <class U> using type2 = decltype([](auto) {}(1));
  template <class U> using type3 = decltype([](True auto) {}(1));
  template <class>
  using type4 = decltype([](auto... pack) { return sizeof...(pack); }(1, 2));

  template <class U> using type5 = decltype([](False auto...) {}(1)); // #Type5

  template <class U>
  using type6 = decltype([]<True> {}.template operator()<char>());
  template <class U>
  using type7 = decltype([]<False> {}.template operator()<char>()); // #Type7

  template <class U>
  using type8 = decltype([]() // #Type8
                           requires(sizeof(U) == 32) // #Type8-requirement
                         {}());

  template <class... U>
  using type9 = decltype([]<True>(U...) {}.template operator()<char>(U()...));
  // https://github.com/llvm/llvm-project/issues/76674
  template <class U>
  using type10 = decltype([]<class V> { return V(); }.template operator()<U>());

  template <class U> using type11 = decltype([] { return U{}; });
};

template <class> using Meow = decltype([]<True> {}.template operator()<int>());

template <class... U>
using MeowMeow = decltype([]<True>(U...) {}.template operator()<char>(U()...));

// https://github.com/llvm/llvm-project/issues/70601
template <class> using U = decltype([]<True> {}.template operator()<int>());

U<int> foo();

void bar() {
  using T = S<int>::type<int, int, int>;
  using T2 = S<int>::type2<int>;
  using T3 = S<int>::type3<char>;
  using T4 = S<int>::type4<void>;
  using T5 = S<int>::type5<void>; // #T5
  // expected-error@#Type5 {{no matching function for call}}
  // expected-note@#T5 {{type alias 'type5' requested here}}
  // expected-note@#Type5 {{constraints not satisfied [with auto:1 = <int>]}}
  // expected-note@#Type5 {{because 'int' does not satisfy 'False'}}
  // expected-note@#False {{because 'false' evaluated to false}}

  using T6 = S<int>::type6<void>;
  using T7 = S<int>::type7<void>; // #T7
  // expected-error@#Type7 {{no matching member function for call}}
  // expected-note@#T7 {{type alias 'type7' requested here}}
  // expected-note@#Type7 {{constraints not satisfied [with $0 = char]}}
  // expected-note@#Type7 {{because 'char' does not satisfy 'False'}}
  // expected-note@#False {{because 'false' evaluated to false}}

  using T8 = S<int>::type8<char>; // #T8
  // expected-error@#Type8 {{no matching function for call}}
  // expected-note@#T8 {{type alias 'type8' requested here}}
  // expected-note@#Type8 {{constraints not satisfied}}
  // expected-note@#Type8-requirement {{because 'sizeof(char) == 32' (1 == 32) evaluated to false}}

  using T9 = S<int>::type9<long, long, char>;
  using T10 = S<int>::type10<int>;
  using T11 = S<int>::type11<int>;
  int x = T11()();
  using T12 = Meow<int>;
  using T13 = MeowMeow<char, int, long, unsigned>;

  static_assert(__is_same(T, void));
  static_assert(__is_same(T2, void));
  static_assert(__is_same(T3, void));
  static_assert(__is_same(T4, decltype(sizeof(0))));
  static_assert(__is_same(T6, void));
  static_assert(__is_same(T9, void));
  static_assert(__is_same(T10, int));
  static_assert(__is_same(T12, void));
  static_assert(__is_same(T13, void));
}

namespace GH82104 {

template <typename, typename...> int Zero = 0;

template <typename T, typename...U>
using T14 = decltype([]<int V = 0>() { return Zero<T, U...>; }());

template <typename T> using T15 = T14<T, T>;

static_assert(__is_same(T15<char>, int));

} // namespace GH82104

} // namespace lambda_calls
