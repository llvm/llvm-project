// RUN: %clang_cc1 -std=c++2a -emit-llvm-only -Wno-unused-value %s -verify

// expected-no-diagnostics

struct A {
  consteval A operator+() { return {}; }
};
consteval A operator~(A) { return {}; }
consteval A operator+(A, A) { return {}; }

template <class> void f() {
  A a;
  A b = ~a;
  A c = a + a;
  A d = +a;
}
template void f<int>();

template <class T> void foo() {
  T a;
  T b = ~a;
  T c = a + a;
  T d = +a;
}

template void foo<A>();

template <typename DataT> struct B { DataT D; };

template <typename DataT>
consteval B<DataT> operator+(B<DataT> lhs, B<DataT> rhs) {
  return B<DataT>{lhs.D + rhs.D};
}

template <class T> consteval T template_add(T a, T b) { return a + b; }

consteval B<int> non_template_add(B<int> a, B<int> b) { return a + b; }

void bar() {
  constexpr B<int> a{};
  constexpr B<int> b{};
  auto constexpr c = a + b;
}

static_assert((template_add(B<int>{7}, B<int>{3})).D == 10);
static_assert((non_template_add(B<int>{7}, B<int>{3})).D == 10);
