// RUN:  %clang_cc1 -std=c++2a -verify %s
template<typename T, typename U> constexpr bool is_same_v = false;
template<typename T> constexpr bool is_same_v<T, T> = true;

template<typename T, unsigned size>
concept LargerThan = sizeof(T) > size;
// expected-note@-1 2{{because 'sizeof(char) > 1U' (1 > 1) evaluated to false}}
// expected-note@-2 {{because 'sizeof(int) > 10U' (4 > 10) evaluated to false}}
// expected-note@-3 {{because 'sizeof(int) > 4U' (4 > 4) evaluated to false}}

template<typename T>
concept Large = LargerThan<T, 1>;
// expected-note@-1 2{{because 'LargerThan<char, 1>' evaluated to false}}

namespace X {
  template<typename T, unsigned size>
  concept SmallerThan = sizeof(T) < size;
  template<typename T>
  concept Small = SmallerThan<T, 2>;
}

Large auto test1() { // expected-error{{deduced type 'char' does not satisfy 'Large'}}
  Large auto i = 'a';
  // expected-error@-1{{deduced type 'char' does not satisfy 'Large'}}
  Large auto j = 10;
  ::Large auto k = 10;
  LargerThan<1> auto l = 10;
  ::LargerThan<1> auto m = 10;
  LargerThan<10> auto n = 10;
  // expected-error@-1{{deduced type 'int' does not satisfy 'LargerThan<10>'}}
  X::Small auto o = 'x';
  X::SmallerThan<5> auto p = 1;
  return 'a';
}

::Large auto test2() { return 10; }
LargerThan<4> auto test3() { return 10; }
// expected-error@-1{{deduced type 'int' does not satisfy 'LargerThan<4>'}}
::LargerThan<2> auto test4() { return 10; }

Large auto test5() -> void;
// expected-error@-1{{function with trailing return type must specify return type 'auto', not 'Large auto'}}
auto test6() -> Large auto { return 1; }

X::Small auto test7() { return 'a'; }
X::SmallerThan<5> auto test8() { return 10; }

namespace PR48384 {
  int a = 0;
  int &b = a;
  template <class T> concept True = true;

  True auto c = a;
  static_assert(is_same_v<decltype(c), int>);

  True auto d = b;
  static_assert(is_same_v<decltype(d), int>);

  True decltype(auto) e = a;
  static_assert(is_same_v<decltype(e), int>);

  True decltype(auto) f = b;
  static_assert(is_same_v<decltype(f), int&>);

  True decltype(auto) g = (a);
  static_assert(is_same_v<decltype(g), int&>);

  True decltype(auto) h = (b);
  static_assert(is_same_v<decltype(h), int&>);
}

namespace PR48593 {
  template <class T, class U> concept a = true;
  a<B> auto c = 0; // expected-error{{use of undeclared identifier 'B'}}

  template<class> concept d = true;
  d<,> auto e = 0; // expected-error{{expected expression}}
}

namespace PR48617 {
  template <typename...> concept C = true;
  template <typename...> class A {};

  template <typename... Ts> C<Ts...> auto e(A<Ts...>) { return 0; }
  template auto e<>(A<>);
  template auto e<int>(A<int>);

  template <typename... Ts> C<Ts...> auto d(A<Ts...>) { return 0; }
  template C<> auto d<>(A<>);
  template C<int> auto d<int>(A<int>);

  template <typename... Ts> A<Ts...> c(Ts...);
  int f = e(c(1, 2));
}
