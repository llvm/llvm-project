// RUN: %clang_cc1 -fsyntax-only -std=c++20 -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c++26 -verify %s

namespace GH214477 {

struct dummy {};
template <typename A> struct real { A a; };

template <typename V> struct Tests {
  template <int> static constexpr dummy obj = {};

  const int reg = [] {
    using X = decltype(obj<0>);
    static_assert(__is_same(X, const real<int>));
    (void)&obj<0>;
    return 0;
  }();

  template <int Tmp>
    requires(Tmp == 0)
  static constexpr real obj<Tmp> = {0};
};

void test() { (void)Tests<int>{}; }

template <typename V> struct Tests2 {
  template <int> static constexpr dummy obj = {};

  void f() {
    using X = decltype(obj<0>);
    static_assert(__is_same(X, const real<int>));
    (void)&obj<0>;
  }

  template <int Tmp>
    requires(Tmp == 0)
  static constexpr real obj<Tmp> = {0};
};

void test2() { Tests2<int>{}.f(); }

template <typename V> struct Tests3 {
  template <int> static constexpr dummy obj = {};

  const int reg = [] {
    (void)&obj<0>;
    using X = decltype(obj<0>);
    static_assert(__is_same(X, const real<int>));
    return 0;
  }();

  template <int Tmp>
    requires(Tmp == 0)
  static constexpr real obj<Tmp> = {0};
};

void test3() { (void)Tests3<int>{}; }

} // namespace GH214477

namespace deduced_type_cycle {

template <int> constexpr auto a = 0;
template <int> constexpr auto b = 0;

template <int N>
  requires(N == 0)
constexpr auto a<N> = b<N>; // expected-note {{in instantiation of variable template specialization 'deduced_type_cycle::b<N>' requested here}}

template <int N>
  requires(N == 0)
constexpr auto b<N> = a<N>; // expected-error {{the type of variable template specialization 'a<0>' declared with deduced type 'const auto' depends on itself}}

auto y = a<0>; // expected-note {{in instantiation of variable template specialization 'deduced_type_cycle::a<N>' requested here}}

} // namespace deduced_type_cycle
