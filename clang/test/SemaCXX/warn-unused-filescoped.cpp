// RUN: %clang_cc1 -fsyntax-only -verify -Wunused -Wunused-template -Wunused-member-function -Wno-unused-local-typedefs \
// RUN:            -Wno-c++11-extensions -Wno-c++14-extensions -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wunused -Wunused-template -Wunused-member-function -Wno-unused-local-typedefs -std=c++14 %s

#ifdef HEADER

static void headerstatic() {} // expected-warning{{unused function 'headerstatic'}}
static inline void headerstaticinline() {}

namespace {
void headeranon() {} // expected-warning{{unused function 'headeranon'}}
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

static void f1(); // expected-warning{{unused function 'f1'}}

namespace {
void f2(); // expected-warning{{unused function 'f2'}}

void f3() {} // expected-warning{{unused function 'f3'}}

struct S {
  void m1() {} // expected-warning{{unused member function 'm1'}}
  void m2();   // expected-warning{{unused member function 'm2'}}
  void m3();
  S(const S &);
  void operator=(const S &);
};

  template <typename T>
  struct TS {
    void m();
  };
  template <> void TS<int>::m() {} // expected-warning{{unused member function 'm'}}

  template <typename T>
  void tf() {}                  // expected-warning{{unused function template 'tf'}}
  template <> void tf<int>() {} // expected-warning{{unused function 'tf<int>'}}

  struct VS {
    virtual void vm() { }
  };
  
  struct SVS : public VS {
    void vm() { }
  };
}

void S::m3() {} // expected-warning{{unused member function 'm3'}}

static inline void f4() {} // expected-warning{{unused function 'f4'}}
const unsigned int cx = 0; // expected-warning{{unused variable 'cx'}}
const unsigned int cy = 0;
int f5() { return cy; }

static int x1; // expected-warning{{unused variable 'x1'}}

namespace {
int x2; // expected-warning{{unused variable 'x2'}}

struct S2 {
  static int x; // expected-warning{{unused variable 'x'}}
};

  template <typename T>
  struct TS2 {
    static int x;
  };
  template <> int TS2<int>::x; // expected-warning{{unused variable 'x'}}

  template <typename T, typename U> int vt = 0; // expected-warning {{unused variable template 'vt'}}
  template <typename T> int vt<T, void> = 0;
  template <> int vt<void, void> = 0; // expected-warning {{unused variable 'vt<void, void>'}}
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

  void test(A a); // expected-warning {{unused function 'test'}}
  extern "C" void test4(A a);
}

namespace rdar8733476 {
static void foo() {}                         // expected-warning {{function 'foo' is not needed and will not be emitted}}
template <typename T> static void foo_t() {} // expected-warning {{unused function template 'foo_t'}}
template <> void foo_t<int>() {}             // expected-warning {{function 'foo_t<int>' is not needed and will not be emitted}}

template <int>
void bar() {
  foo();
  foo_t<int>();
  foo_t<void>();
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
  static const double d = 0.0; // expected-warning{{variable 'd' is not needed and will not be emitted}}
  int y = sizeof(d);

  namespace {
  // FIXME: Should be "unused variable template 'var_t'" instead.
  template <typename T> const double var_t = 0; // expected-warning {{unused variable 'var_t'}}
  template <> const double var_t<int> = 0;      // expected-warning {{variable 'var_t<int>' is not needed and will not be emitted}}
  int z = sizeof(var_t<int>);                   // expected-warning {{unused variable 'z'}}
  }                                             // namespace
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
    void func() { // expected-warning {{unused member function 'func'}}
    }
  } x; // expected-warning {{unused variable 'x'}}
}

namespace test6 {
  typedef struct { // expected-warning {{add a tag name}}
    void bar(); // expected-note {{}}
  } A; // expected-note {{}}

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
template <typename T>
static void completeRedeclChainForTemplateSpecialization() {} // expected-warning {{unused function template 'completeRedeclChainForTemplateSpecialization'}}
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
static constexpr int constexpr3() { return 1; } // expected-warning {{unused function 'constexpr3'}}
constexpr int constexpr4() { return 2; }
#endif
}

#endif
