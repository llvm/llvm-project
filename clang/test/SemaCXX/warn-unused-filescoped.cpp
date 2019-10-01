// RUN: %clang_cc1 -fsyntax-only -verify -Wunused -Wunused-template -Wunused-member-function -Wno-unused-local-typedefs -Wno-c++11-extensions -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wunused -Wunused-template -Wunused-member-function -Wno-unused-local-typedefs -std=c++14 %s

#ifdef HEADER

static void headerstatic() {}  // expected-warning{{unused}}
static inline void headerstaticinline() {}

namespace {
  void headeranon() {}  // expected-warning{{unused}}
  inline void headerinlineanon() {}
}

namespace test7
{
  template<typename T>
  static inline void foo(T) { }

  // This should not emit an unused-function warning since it inherits
  // the static storage type from the base template.
  template<>
  inline void foo(int) {  }

  // Partial specialization
  template<typename T, typename U>
  static inline void bar(T, U) { }

  template<typename U>
  inline void bar(int, U) { }

  template<>
  inline void bar(int, int) { }
};

namespace pr19713 {
#if __cplusplus >= 201103L
  static constexpr int constexpr1() { return 1; }
  constexpr int constexpr2() { return 2; }
#endif
}

#else
#define HEADER
#include "warn-unused-filescoped.cpp"

static void f1(); // expected-warning{{unused}}

namespace {
  void f2();  // expected-warning{{unused}}

  void f3() { }  // expected-warning{{unused}}

  struct S {
    void m1() { }  // expected-warning{{unused}}
    void m2();  // expected-warning{{unused}}
    void m3();
    S(const S&);
    void operator=(const S&);
  };

  template <typename T>
  struct TS {
    void m();
  };
  template <> void TS<int>::m() { }  // expected-warning{{unused}}

  template <typename T>
  void tf() { }  // expected-warning{{unused}}
  template <> void tf<int>() { }  // expected-warning{{unused}}
  
  struct VS {
    virtual void vm() { }
  };
  
  struct SVS : public VS {
    void vm() { }
  };
}

void S::m3() { }  // expected-warning{{unused}}

static inline void f4() { }  // expected-warning{{unused}}
const unsigned int cx = 0; // expected-warning{{unused}}
const unsigned int cy = 0;
int f5() { return cy; }

static int x1;  // expected-warning{{unused}}

namespace {
  int x2;  // expected-warning{{unused}}
  
  struct S2 {
    static int x;  // expected-warning{{unused}}
  };

  template <typename T>
  struct TS2 {
    static int x;
  };
  template <> int TS2<int>::x;  // expected-warning{{unused}}
}

namespace PR8841 {
  // Ensure that friends of class templates are considered to have a dependent
  // context and not marked unused.
  namespace {
    template <typename T> struct X {
      friend bool operator==(const X&, const X&) { return false; }
    };
  }
  template <typename T> void template_test(X<T> x) {
    (void)(x == x);
  }
  void test() {
    X<int> x;
    template_test(x);
  }
}

namespace test4 {
  namespace { struct A {}; }

  void test(A a); // expected-warning {{unused function}}
  extern "C" void test4(A a);
}

namespace rdar8733476 {
  static void foo() { } // expected-warning {{not needed and will not be emitted}}

  template <int>
  void bar() {
    foo();
  }
}

namespace test5 {
  static int n = 0;
  static int &r = n;
  int f(int &);
  int k = f(r);

  // FIXME: We should produce warnings for both of these.
  static const int m = n;
  int x = sizeof(m);
  static const double d = 0.0; // expected-warning{{not needed and will not be emitted}}
  int y = sizeof(d);
}

namespace unused_nested {
  class outer {
    void func1();
    struct {
      void func2() {
      }
    } x;
  };
}

namespace unused {
  struct {
    void func() { // expected-warning {{unused member function}}
    }
  } x; // expected-warning {{unused variable}}
}

namespace test6 {
  typedef struct {
    void bar();
  } A;

  typedef struct {
    void bar();  // expected-warning {{unused member function 'bar'}}
  } *B;

  struct C {
    void bar();
  };
}

namespace pr14776 {
  namespace {
    struct X {};
  }
  X a = X(); // expected-warning {{unused variable 'a'}}
  auto b = X(); // expected-warning {{unused variable 'b'}}
}

namespace UndefinedInternalStaticMember {
  namespace {
    struct X {
      static const unsigned x = 3;
      int y[x];
    };
  }
}

namespace test8 {
static void func();
void bar() { void func() __attribute__((used)); }
static void func() {}
}

namespace test9 {
template<typename T>
static void completeRedeclChainForTemplateSpecialization() { } // expected-warning {{unused}}
}

namespace test10 {
#if __cplusplus >= 201103L
// FIXME: Warn on template definitions with no instantiations?
template<class T>
constexpr T pi = T(3.14);
#endif
}

namespace pr19713 {
#if __cplusplus >= 201103L
  // FIXME: We should warn on both of these.
  static constexpr int constexpr3() { return 1; } // expected-warning {{unused}}
  constexpr int constexpr4() { return 2; }
#endif
}

#endif
