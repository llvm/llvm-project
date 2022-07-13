// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace test0 {
  struct A { // expected-note {{candidate function (the implicit copy assignment operator) not viable: 'this' argument has type 'const test0::A', but method is not marked const}}
#if __cplusplus >= 201103L
  // expected-note@-2 {{candidate function (the implicit move assignment operator) not viable: 'this' argument has type 'const test0::A', but method is not marked const}}
#endif
    A &operator=(void*); // expected-note {{candidate function not viable: 'this' argument has type 'const test0::A', but method is not marked const}}
  };

  void test(const A &a) {
    a = "help"; // expected-error {{no viable overloaded '='}}
  }
}

namespace PR16314 {
  void f(char*);
  int &f(...);
  void x()
  {
    int &n = f("foo");
#if __cplusplus < 201103L
    // expected-warning@-2 {{conversion from string literal to 'char *' is deprecated}}
    // expected-error@-3 {{non-const lvalue reference to type 'int' cannot bind to a temporary of type 'void'}}
#endif
  }
}

namespace warn_if_best {
  int f(char *);
  void f(double);
  void x()
  {
    int n = f("foo");
#if __cplusplus < 201103L
    // expected-warning@-2 {{conversion from string literal to 'char *' is deprecated}}
#else
    // expected-warning@-4 {{ISO C++11 does not allow conversion from string literal to 'char *'}}
#endif
  }
}

namespace userdefined_vs_illformed {
  struct X { X(const char *); };

  void *f(char *p); // best for C++03
  double f(X x);  // best for C++11
  void g()
  {
    double d = f("foo");
#if __cplusplus < 201103L
    // expected-warning@-2 {{conversion from string literal to 'char *' is deprecated}}
    // expected-error@-3 {{cannot initialize a variable of type 'double' with an rvalue of type 'void *'}}
#endif
  }
}

namespace sfinae_test {
  int f(int, char*);

  template<int T>
  struct S { typedef int type; };

  template<>
  struct S<sizeof(int)> { typedef void type; };

  // C++11: SFINAE failure
  // C++03: ok
  template<typename T> int cxx11_ignored(T, typename S<sizeof(f(T(), "foo"))>::type *);
#if __cplusplus < 201103L
  // expected-warning@-2 {{conversion from string literal to 'char *' is deprecated}}
#else
  // expected-note@-4 {{candidate template ignored: substitution failure}}
#endif

  // C++11: better than latter
  // C++03: worse than latter
  template<typename T> void g(T, ...);
  template<typename T> int g(T, typename S<sizeof(f(T(), "foo"))>::type *);
#if __cplusplus < 201103L
  // expected-warning@-2 {{conversion from string literal to 'char *' is deprecated}}
#endif

  int a = cxx11_ignored(0, 0);
  int b = g(0, 0);
#if __cplusplus >= 201103L
  // expected-error@-3 {{no matching function for call to 'cxx11_ignored'}}
  // expected-error@-3 {{cannot initialize a variable of type 'int' with an rvalue of type 'void'}}
#endif
}
