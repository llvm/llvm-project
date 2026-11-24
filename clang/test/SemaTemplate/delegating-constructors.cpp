// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -verify 

namespace PR10457 {

  class string
  {
    string(const char* str, unsigned);

  public:
    template <unsigned N>
    string(const char (&str)[N])
      : string(str) {} // expected-error{{constructor for 'string<6U>' creates a delegation cycle}}
  };

  void f() {
    string s("hello");
  }

  struct Foo {
   Foo(int) { }


   template <typename T>
   Foo(T, int i) : Foo(i) { }
};

  void test_Foo()
  {
    Foo f(1, 1);
  }
}

namespace PR12890 {
  class Document
  {
  public:
      Document() = default;

      template <class T>
      explicit
      Document(T&& t) : Document()
      {
      }
  };
  void f()
  {
      Document d(1);
  }
}

namespace GH239861 {
  template <class T> struct S {
    using type = int;
  };

  template <class T> T &&foo(typename S<T>::type &t);

  template <typename... Args> struct F {};

  struct D1 {
    D1(int);
    template <typename... Args>
    D1(Args &&...args) : D1(foo<Args>(args)...) {}
  };

  void test_D1() {
    int x = 42;
    D1 d(x);
  }

  struct D2 {
    template <typename... Args>
    D2(Args &&...args) : D2(foo<Args>(args)...), F<Args...>() {} // expected-error {{an initializer for a delegating constructor must appear alone}}
  };

  struct D3 {
    int x;
    template <typename... Args>
    D3(Args &&...args) : D3(foo<Args>(args)...), x(0) {} // expected-error {{an initializer for a delegating constructor must appear alone}}
  };

  template <typename T>
  struct D4 {
    D4(int);
    template <typename U>
    D4(U u) : D4(foo<U>(u)) {}
  };

  void test_D4() {
    int x = 42;
    D4<int> d(x);
  }

  template <typename T>
  struct D5 {
    int x;
    template <typename U>
    D5(U u) : D5(foo<U>(u)), x(0) {} // expected-error {{an initializer for a delegating constructor must appear alone}}
  };
}
