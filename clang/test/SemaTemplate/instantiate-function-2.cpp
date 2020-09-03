// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

template <typename T> struct S {
  S() { }
  S(T t);
};

template struct S<int>;

void f() {
  S<int> s1;
  S<int> s2(10);
}

namespace PR7184 {
  template<typename T>
  void f() {
    typedef T type;
    void g(int array[sizeof(type)]);
  }

  template void f<int>();
}

namespace UsedAttr {
  template<typename T>
  void __attribute__((used)) foo() {
    T *x = 1; // expected-error{{cannot initialize a variable of type 'int *' with an rvalue of type 'int'}}
  }

  void bar() {
    foo<int>(); // expected-note{{instantiation of}}
  }
}

namespace PR9654 {
  typedef void ftype(int);

  template<typename T>
  ftype f;

  void g() {
    f<int>(0);
  }
}

namespace AliasTagDef {
  template<typename T>
  T f() {
    using S = struct { // expected-warning {{add a tag name}} expected-note {{}}
#if __cplusplus <= 199711L
    // expected-warning@-2 {{alias declarations are a C++11 extension}}
#endif
      T g() { // expected-note {{}}
        return T();
      }
    };
    return S().g();
  }

  int n = f<int>();
}

namespace PR10273 {
  template<typename T> void (f)(T t) {}

  void g() {
    (f)(17);
  }
}

namespace rdar15464547 {
  class A {
    A();
  };

  template <typename R> class B {
  public:
    static void meth1();
    static void meth2();
  };

  A::A() {
    extern int compile_time_assert_failed;
    B<int>::meth2();
  }

  template <typename R> void B<R>::meth1() {
    extern int compile_time_assert_failed;
  }

  template <typename R> void B<R>::meth2() {
    extern int compile_time_assert_failed;
  }
}
