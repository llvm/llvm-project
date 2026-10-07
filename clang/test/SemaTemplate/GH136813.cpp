// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

namespace GH136813 {

template <typename, typename>
class C {
public:
  template <typename>
  class C2 { // expected-note {{candidate template ignored}} \
             // expected-note {{implicit deduction guide declared as}}
  public:
    C2(C &) {} // expected-note {{candidate template ignored}} \
                // expected-note {{implicit deduction guide declared as}}
  };

  void f() {
    C2(*this); // expected-error {{no viable constructor or deduction guide}}
  }
};

void test_original() {
  C<int, int>().f(); // expected-note {{in instantiation of member function}}
}

namespace DeducibleCases {

template <class T>
struct Outer1 {
  template <class U>
  struct Inner {
    Inner(Outer1, U) {}
  };

  void f() { Inner(*this, 42); }
};

template <class T>
struct Outer2 {
  template <class U>
  struct Inner {
    Inner(Outer2 *, U) {}
  };

  void f() { Inner(this, 42); }
};

template <class T>
struct Outer3 {
  template <class U>
  struct Inner {
    Inner(const Outer3 &, U) {}
  };

  void f() { Inner(*this, 42); }
};

template <class T>
struct Outer4 {
  template <class U>
  struct Inner {
    Inner(Outer4 &&, U) {}
  };

  void f() { Inner(Outer4{}, 42); }
};

void test() {
  Outer1<int>().f();
  Outer2<int>().f();
  Outer3<int>().f();
  Outer4<int>().f();
}

} // namespace DeducibleCases
} // namespace GH136813
