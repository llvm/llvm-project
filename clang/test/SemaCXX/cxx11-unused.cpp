// RUN: %clang_cc1 -std=c++11 -verify %s -Wunused-parameter

// PR19303 : Make sure we don't get a unused expression warning for deleted and
// defaulted functions

// expected-no-diagnostics

class A {
public:
  int x;
  A() = default;
  ~A() = default;
  A(const A &other) = delete;

  template <typename T>
  void SetX(T x) {
    this->x = x;
  };

  void SetX1(int x);
};

template <>
void A::SetX(A x) = delete;

class B {
public:
  B() = default;
  ~B() = default;
  B(const B &other);
};

B::B(const B &other) = default;
