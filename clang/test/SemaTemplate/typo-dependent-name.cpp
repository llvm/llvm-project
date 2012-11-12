// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

template<typename T>
struct Base {
  T inner;
};

template<typename T>
struct X {
  template<typename U>
  struct Inner {
  };

  bool f(T other) {
    return this->inner < other;
  }
};
