// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s
// expected-no-diagnostics

template<typename T>
struct A {
  template<typename U>
  struct B {
    static constexpr int y = 0;
  };

  template<typename U>
  struct B<U*> {
    static constexpr int y = 1;
  };

  template<typename U>
  static constexpr int x = 0;

  template<typename U>
  static constexpr int x<U*> = 1;
};

static_assert(A<short>::B<int>::y == 0);
static_assert(A<short>::B<int*>::y == 1);
static_assert(A<short>::x<int> == 0);
static_assert(A<short>::x<int*> == 1);

template<>
template<typename U>
struct A<long>::B {
  static constexpr int y = 2;
};

template<>
template<typename U>
struct A<long>::B<U&> {
  static constexpr int y = 3;
};

template<>
template<typename U>
constexpr int A<long>::x = 2;

template<>
template<typename U>
constexpr int A<long>::x<U&> = 3;

static_assert(A<long>::B<int>::y == 2);
static_assert(A<long>::B<int*>::y == 2);
static_assert(A<long>::B<int&>::y == 3);
static_assert(A<long>::x<int> == 2);
static_assert(A<long>::x<int*> == 2);
static_assert(A<long>::x<int&> == 3);
