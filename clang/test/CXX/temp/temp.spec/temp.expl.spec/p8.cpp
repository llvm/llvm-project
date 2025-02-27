// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s
// expected-no-diagnostics

template<typename T>
concept C = sizeof(T) <= sizeof(long);

template<typename T>
struct A {
  template<typename U>
  void f(U) requires C<U>;

  void g() requires C<T>;

  template<typename U>
  void h(U) requires C<T>;

  constexpr int i() requires C<T> {
    return 0;
  }

  constexpr int i() requires C<T> && true {
    return 1;
  }

  template<>
  void f(char);
};

template<>
template<typename U>
void A<short>::f(U) requires C<U>;

template<>
template<typename U>
void A<short>::h(U) requires C<short>;

template<>
template<>
void A<int>::f(int);

template<>
void A<long>::g();

template<>
constexpr int A<long>::i() {
  return 2;
}

static_assert(A<long>().i() == 2);

template<typename T>
struct D {
  template<typename U>
  static constexpr int f(U);

  template<typename U>
  static constexpr int f(U) requires (sizeof(T) == 1);

  template<>
  constexpr int f(int) {
    return 1;
  }
};

template<>
template<typename U>
constexpr int D<signed char>::f(U) requires (sizeof(signed char) == 1) {
  return 0;
}

static_assert(D<char>::f(0) == 1);
static_assert(D<char[2]>::f(0) == 1);
static_assert(D<signed char>::f(0) == 1);
static_assert(D<signed char>::f(0.0) == 0);
