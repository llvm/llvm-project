// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -verify 

namespace PR10457 {

  class string
  {
    string(const char* str, unsigned);

  public:
    template <unsigned N>
    string(const char (&str)[N])
      : string(str) {} // expected-error{{constructor for 'string<6>' creates a delegation cycle}}
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
