// RUN: %clang_cc1 -std=c++98 %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify=expected,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 %s -verify=expected,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 %s -verify=expected,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++20 %s -verify=expected,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++23 %s -verify=expected,since-cxx11 -fexceptions -fcxx-exceptions -pedantic-errors

namespace std {
  __extension__ typedef __SIZE_TYPE__ size_t;

  template<typename T> struct initializer_list {
    const T *p; size_t n;
    initializer_list(const T *p, size_t n);
  };
}

namespace cwg930 { // cwg930: 2.7
#if __cplusplus >= 201103L
static_assert(alignof(int[]) == alignof(int), "");
static_assert(alignof(int[][2]) == alignof(int[2]), "");
#endif
} // namespace cwg930

namespace cwg948 { // cwg948: 3.7
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

namespace cwg952 { // cwg952: 2.8
namespace example1 {
struct A {
  typedef int I; // #cwg952-I
};
struct B : private A { // #cwg952-B
};
struct C : B {
  void f() {
    I i1;
    // expected-error@-1 {{'I' is a private member of 'cwg952::example1::A'}}
    //   expected-note@#cwg952-B {{constrained by private inheritance here}}
    //   expected-note@#cwg952-I {{member is declared here}}
  }
  I i2;
  // expected-error@-1 {{'I' is a private member of 'cwg952::example1::A'}}
  //   expected-note@#cwg952-B {{constrained by private inheritance here}}
  //   expected-note@#cwg952-I {{member is declared here}}
  struct D {
    I i3;
    // expected-error@-1 {{'I' is a private member of 'cwg952::example1::A'}}
    //   expected-note@#cwg952-B {{constrained by private inheritance here}}
    //   expected-note@#cwg952-I {{member is declared here}}
    void g() {
      I i4;
      // expected-error@-1 {{'I' is a private member of 'cwg952::example1::A'}}
      //   expected-note@#cwg952-B {{constrained by private inheritance here}}
      //   expected-note@#cwg952-I {{member is declared here}}
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
} // namespace cwg952

namespace cwg974 { // cwg974: yes
#if __cplusplus >= 201103L
  void test() {
    auto lam = [](int x = 42) { return x; };
  }
#endif
}

namespace cwg977 { // cwg977: yes
enum E { e = E() }; // #cwg977-E
#if !defined(_WIN32) || defined(__MINGW32__)
// expected-error@#cwg977-E {{invalid use of incomplete type 'E'}}
//   expected-note@#cwg977-E {{definition of 'cwg977::E' is not complete until the closing '}'}}
#endif
#if __cplusplus >= 201103L
enum E2 : int { e2 = E2() };
enum struct E3 { e = static_cast<int>(E3()) };
enum struct E4 : int { e = static_cast<int>(E4()) };
#endif
} // namespace cwg977

namespace cwg990 { // cwg990: 3.5
#if __cplusplus >= 201103L
  struct A { // #cwg990-A
    A(std::initializer_list<int>); // #cwg990-A-init-list
  };
  struct B {
    A a;
  };
  B b1 { };
  B b2 { 1 };
  // since-cxx11-error@-1 {{no viable conversion from 'int' to 'A'}}
  //   since-cxx11-note@#cwg990-A {{candidate constructor (the implicit copy constructor) not viable: no known conversion from 'int' to 'const A &' for 1st argument}}
  //   since-cxx11-note@#cwg990-A {{candidate constructor (the implicit move constructor) not viable: no known conversion from 'int' to 'A &&' for 1st argument}}
  //   since-cxx11-note@#cwg990-A-init-list {{candidate constructor not viable: no known conversion from 'int' to 'std::initializer_list<int>' for 1st argument}}
  B b3 { { 1 } };

  struct C {
    C();
    C(int);
    C(std::initializer_list<int>) = delete; // #cwg990-deleted
  };
  C c1[3] { 1 }; // ok
  C c2[3] { 1, {2} };
  // since-cxx11-error@-1 {{call to deleted constructor of 'C'}}
  //   since-cxx11-note@#cwg990-deleted {{'C' has been explicitly marked deleted here}}

  struct D {
    D();
    D(std::initializer_list<int>);
    D(std::initializer_list<double>);
  };
  D d{};
#endif
}
