// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++20 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++2b %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

namespace std {
  __extension__ typedef __SIZE_TYPE__ size_t;

  template<typename T> struct initializer_list {
    const T *p; size_t n;
    initializer_list(const T *p, size_t n);
  };
}

namespace dr990 { // dr990: 3.5
#if __cplusplus >= 201103L
  struct A { // expected-note 2{{candidate}}
    A(std::initializer_list<int>); // expected-note {{candidate}}
  };
  struct B {
    A a;
  };
  B b1 { };
  B b2 { 1 }; // expected-error {{no viable conversion from 'int' to 'A'}}
  B b3 { { 1 } };

  struct C {
    C();
    C(int);
    C(std::initializer_list<int>) = delete; // expected-note {{here}}
  };
  C c1[3] { 1 }; // ok
  C c2[3] { 1, {2} }; // expected-error {{call to deleted}}

  struct D {
    D();
    D(std::initializer_list<int>);
    D(std::initializer_list<double>);
  };
  D d{};
#endif
}

namespace dr948 { // dr948: 3.7
#if __cplusplus >= 201103L
  class A {
  public:
     constexpr A(int v) : v(v) { }
     constexpr operator int() const { return v; }
  private:
     int v;
  };

  constexpr int id(int x)
  {
    return x;
  }

  void f() {
     if (constexpr int i = id(101)) { }
     switch (constexpr int i = id(2)) { default: break; case 2: break; }
     for (; constexpr int i = id(0); ) { }
     while (constexpr int i = id(0)) { }

     if (constexpr A i = 101) { }
     switch (constexpr A i = 2) { default: break; case 2: break; }
     for (; constexpr A i = 0; ) { }
     while (constexpr A i = 0) { }
  }
#endif
}

namespace dr952 { // dr952: yes
namespace example1 {
struct A {
  typedef int I; // #dr952-typedef-decl
};
struct B : private A { // #dr952-inheritance
};
struct C : B {
  void f() {
    I i1; // expected-error {{private member}}
    // expected-note@#dr952-inheritance {{constrained by private inheritance}}
    // expected-note@#dr952-typedef-decl {{declared here}}
  }
  I i2; // expected-error {{private member}}
  // expected-note@#dr952-inheritance {{constrained by private inheritance}}
  // expected-note@#dr952-typedef-decl {{declared here}}
  struct D {
    I i3; // expected-error {{private member}}
    // expected-note@#dr952-inheritance {{constrained by private inheritance}}
    // expected-note@#dr952-typedef-decl {{declared here}}
    void g() {
      I i4; // expected-error {{private member}}
      // expected-note@#dr952-inheritance {{constrained by private inheritance}}
      // expected-note@#dr952-typedef-decl {{declared here}}
    }
  };
};
} // namespace example1
namespace example2 {
struct A {
protected:
  static int x;
};
struct B : A {
  friend int get(B) { return x; }
};
} // namespace example2
} // namespace dr952

namespace dr974 { // dr974: yes
#if __cplusplus >= 201103L
  void test() {
    auto lam = [](int x = 42) { return x; };
  }
#endif
}
