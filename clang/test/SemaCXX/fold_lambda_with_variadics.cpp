// RUN: %clang_cc1 -fsyntax-only -std=c++20 -verify %s

namespace GH85667 {

template <class T>
struct identity {
  using type = T;
};

template <class = void> void f() {

  static_assert([]<class... Is>(Is... x) {
    return ([I(x)] {
      return I;
    }() + ...);
  }(1, 2) == 3);

  []<class... Is>(Is... x) {
    return ([](auto y = Is()) { return y + 1; }() + ...); // expected-error {{no matching function}}                     \
                                                          // expected-note {{couldn't infer template argument 'y:auto'}} \
                                                          // expected-note@-1 {{requested here}}
                                                          // expected-note@#instantiate-f {{requested here}}
  }(1);

  []<class... Is>() {
    ([]<class = Is>(Is)
       noexcept(bool(Is()))
     {}(Is()),
     ...);
  }.template operator()<char, int, float>();

  static_assert(__is_same(decltype([]<class... Is>() {
                            return ([]() -> decltype(Is()) { return {}; }(),
                                    ...);
                          }.template operator()<int, char>()),
                          char));

  []<class... Is>() {
    return ([]<class... Ts>() -> decltype(Is()) { return Ts(); }() + ...);
    // expected-error@-1 {{unexpanded parameter pack 'Ts'}}
  }.template operator()<int, int>();

  // https://github.com/llvm/llvm-project/issues/56852
  []<class... Is>(Is...) {
    ([] {
      using T = identity<Is>::type;
    }(), ...);
  }(1, 2);

  [](auto ...y) {
    ([y] { }(), ...);
  }();

  [](auto ...x) {
    ([&](auto ...y) {
      ([x..., y] { }(), ...);
    })(1);
  }(2, 'b');

  // https://github.com/llvm/llvm-project/issues/18873
  static_assert([]<auto... z>(auto ...x) {
    return [&](auto ...y) {
      return ([x, y] {
        return x + y + z;
      }() + ...);
    }(1, 'a');
  }.template operator()<2, 'b'>(3, 'c') == 1 + 'a' + 2 + 'b' + 3 + 'c');

  [](auto ...x) {  // #outer
    ([&](auto ...y) { // #inner
      ([x, y] { }(), ...);
      // expected-error@-1 {{parameter pack 'y' that has a different length (4 vs. 3) from outer parameter packs}}
      // expected-note-re@#inner {{function template specialization {{.*}} requested here}}
      // expected-note-re@#outer {{function template specialization {{.*}} requested here}}
      // expected-note-re@#instantiate-f {{function template specialization {{.*}} requested here}}
    })('a', 'b', 'c');
  }(0, 1, 2, 3);
}

template void f(); // #instantiate-f

} // namespace GH85667

namespace GH99877 {

struct tuple {
  int x[3];
};

template <class F> int apply(F f, tuple v) { return f(v.x[0], v.x[1], v.x[2]); }

int Cartesian1(auto x, auto y) {
  return apply(
      [&](auto... xs) {
        return (apply([xs](auto... ys) { return (ys + ...); }, y) + ...);
      },
      x);
}

int Cartesian2(auto x, auto y) {
  return apply(
      [&](auto... xs) {
        return (apply([zs = xs](auto... ys) { return (ys + ...); }, y) + ...);
      },
      x);
}

template <int...> struct Ints {};
template <int> struct Choose {
  template <class> struct Templ;
};
template <int... x> int Cartesian3(auto y) {
  return [&]<int... xs>(Ints<xs...>) {
    // check in default template arguments for
    // - type template parameters,
    (void)(apply([]<class = decltype(xs)>(auto... ys) { return (ys + ...); },
                 y) +
           ...);
    // - template template parameters.
    (void)(apply([]<template <class> class = Choose<xs>::template Templ>(
                     auto... ys) { return (ys + ...); },
                 y) +
           ...);
    // - non-type template parameters,
    return (apply([]<int = xs>(auto... ys) { return (ys + ...); }, y) + ...);
  }(Ints<x...>());
}

template <int... x> int Cartesian4(auto y) {
  return [&]<int... xs>(Ints<xs...>) {
    return (
        apply([]<decltype(xs) xx = 1>(auto... ys) { return (ys + ...); }, y) +
        ...);
  }(Ints<x...>());
}

// FIXME: Attributes should preserve the ContainsUnexpandedPack flag.
#if 0

int Cartesian5(auto x, auto y) {
  return apply(
      [&](auto... xs) {
        return (apply([](auto... ys) __attribute__((
                          diagnose_if(!__is_same(decltype(xs), int), "message",
                                      "error"))) { return (ys + ...); },
                      y) +
                ...);
      },
      x);
}

#endif

void foo() {
  auto x = tuple({1, 2, 3});
  auto y = tuple({4, 5, 6});
  Cartesian1(x, y);
  Cartesian2(x, y);
  Cartesian3<1, 2, 3>(y);
  Cartesian4<1, 2, 3>(y);
#if 0
  Cartesian5(x, y);
#endif
}

} // namespace GH99877

namespace GH101754 {

template <typename... Ts> struct Overloaded : Ts... {
  using Ts::operator()...;
};

template <typename... Ts> Overloaded(Ts...) -> Overloaded<Ts...>;

template <class T, class U>
concept same_as = __is_same(T, U);  // #same_as

template <typename... Ts> constexpr auto foo() {
  return Overloaded{[](same_as<Ts> auto value) { return value; }...}; // #lambda
}

static_assert(foo<int, double>()(123) == 123);
static_assert(foo<int, double>()(2.718) == 2.718);

static_assert(foo<int, double>()('c'));
// expected-error@-1 {{no matching function}}

// expected-note@#lambda {{constraints not satisfied}}
// expected-note@#lambda {{'same_as<char, int>' evaluated to false}}
// expected-note@#same_as {{evaluated to false}}

// expected-note@#lambda {{constraints not satisfied}}
// expected-note@#lambda {{'same_as<char, double>' evaluated to false}}
// expected-note@#same_as {{evaluated to false}}

template <class T, class U, class V>
concept C = same_as<T, U> && same_as<U, V>; // #C

template <typename... Ts> constexpr auto bar() {
  return ([]<class Up>() {
    return Overloaded{[](C<Up, Ts> auto value) { // #bar
      return value;
    }...};
  }.template operator()<Ts>(), ...);
}
static_assert(bar<int, float>()(3.14f)); // OK, bar() returns the last overload i.e. <float>.

static_assert(bar<int, float>()(123));
// expected-error@-1 {{no matching function}}
// expected-note@#bar {{constraints not satisfied}}
// expected-note@#bar {{'C<int, float, int>' evaluated to false}}
// expected-note@#C {{evaluated to false}}

// expected-note@#bar {{constraints not satisfied}}
// expected-note@#bar {{'C<int, float, float>' evaluated to false}}
// expected-note@#C {{evaluated to false}}
// expected-note@#same_as 2{{evaluated to false}}

} // namespace GH101754
