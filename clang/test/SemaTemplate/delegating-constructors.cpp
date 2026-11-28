// RUN: %clang_cc1 -fsyntax-only -std=c++03 %s -verify=expected,cxx03
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
      Document() = default; // cxx03-warning {{defaulted function definitions are a C++11 extension}}

      template <class T>
      explicit
      Document(T&& t) : Document() // cxx03-warning {{rvalue references are a C++11 extension}}
      {
      }
  };
  void f()
  {
      Document d(1);
  }
}
