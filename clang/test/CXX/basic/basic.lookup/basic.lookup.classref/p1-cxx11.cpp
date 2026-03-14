// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

template<typename T>
struct set{};
struct Value {
  template<typename T>
  void set(T value) {}

  void resolves_to_same() {
    Value v;
    v.set<double>(3.2);
  }
};
void resolves_to_different() {
  {
    Value v;
    // The fact that the next line is a warning rather than an error is an
    // extension.
    v.set<double>(3.2);
  }
  {
    int set;  // Non-template.
    Value v;
    v.set<double>(3.2);
  }
}

namespace rdar9915664 {
  struct A {
    template<typename T> void a();
  };

  struct B : A { };

  struct C : A { };

  struct D : B, C {
    A &getA() { return static_cast<B&>(*this); }

    void test_a() {
      getA().a<int>();
    }
  };
}

namespace PR11856 {
  template<typename T> T end(T);

  template <typename T>
  void Foo() {
    T it1;
    if (it1->end < it1->end) {
    }
  }

  template<typename T> T *end(T*);

  class X { };
  template <typename T>
  void Foo2() {
    T it1;
    if (it1->end < it1->end) {
    }

    X *x;
    if (x->end < 7) {  // expected-error{{no member named 'end' in 'PR11856::X'}}
    }
  }
}
