// RUN: %clang_cc1 -fsyntax-only -std=c++17 -verify %s -fcxx-exceptions
template<typename T>
struct X0 {
  typedef T* type;

  void f0(T);
  void f1(type);
};

template<> void X0<char>::f0(char);
template<> void X0<char>::f1(type);

namespace PR6161 {
  template<typename _CharT>
  class numpunct : public locale::facet // expected-error{{use of undeclared identifier 'locale'}} \
              // expected-error{{expected class name}}
  {
    static locale::id id; // expected-error{{use of undeclared identifier}}
  };
  numpunct<char>::~numpunct();
}

namespace PR12331 {
  template<typename T> struct S {
    struct U { static const int n = 5; };
    enum E { e = U::n }; // expected-note {{implicit instantiation first required here}}
    int arr[e];
  };
  template<> struct S<int>::U { static const int n = sizeof(int); }; // expected-error {{explicit specialization of 'U' after instantiation}}
}

namespace PR18246 {
  template<typename T>
  class Baz {
  public:
    template<int N> void bar();
  };

  template<typename T>
  template<int N>
  void Baz<T>::bar() {
  }

  template<typename T>
  void Baz<T>::bar<0>() { // expected-error {{cannot specialize a member of an unspecialized template}}
  }
}

namespace PR19340 {
template<typename T> struct Helper {
  template<int N> static void func(const T *m) {}
};

template<typename T> void Helper<T>::func<2>() {} // expected-error {{cannot specialize a member}}
}

namespace SpecLoc {
  template <typename T> struct A {
    static int n; // expected-note {{previous}}
    static void f(); // expected-note {{previous}}
  };
  template<> float A<int>::n; // expected-error {{different type}}
  template<> void A<int>::f() throw(); // expected-error {{does not match}}
}

namespace PR41607 {
  template<int N> struct Outer {
    template<typename...> struct Inner;
    template<> struct Inner<> {
      static constexpr int f() { return N; }
    };

    template<typename...> static int a;
    template<> constexpr int a<> = N;

    template<typename...> static inline int b;
    template<> inline constexpr int b<> = N;

    template<typename...> static constexpr int f();
    template<> constexpr int f() {
      return N;
    }
  };
  static_assert(Outer<123>::Inner<>::f() == 123, "");
  static_assert(Outer<123>::Inner<>::f() != 125, "");

  static_assert(Outer<123>::a<> == 123, "");
  static_assert(Outer<123>::a<> != 125, "");

  static_assert(Outer<123>::b<> == 123, "");
  static_assert(Outer<123>::b<> != 125, "");

  static_assert(Outer<123>::f<>() == 123, "");
  static_assert(Outer<123>::f<>() != 125, "");
}
