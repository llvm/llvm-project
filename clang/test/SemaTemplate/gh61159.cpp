// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s
// expected-no-diagnostics

namespace GH61159 {
template <typename T> struct X {
  struct I;
};

template <> struct X<int>::I {
  template <int ct> constexpr int f() { return ct; };

  int data = 3;
};

template <typename T> struct X<T>::I {
  template <T ct> constexpr T f() { return ct + 1; };
  T data = 7;
};

static_assert(X<int>::I{}.f<17>() == 17);
static_assert(X<int>::I{}.data == 3);
static_assert(X<short>::I{}.data == 7);
static_assert(X<short>::I{}.f<18>() == 19);

template <typename T> struct Y {
  struct I;
};

template <> struct Y<int> {
  struct I {
    template <int ct> constexpr int f() { return ct; };
    int data = 3;
  };
};

static_assert(Y<int>::I{}.f<17>() == 17);
static_assert(Y<int>::I{}.data == 3);

} // namespace GH61159
