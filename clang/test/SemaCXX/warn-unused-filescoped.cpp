// RUN: %clang_cc1 -fsyntax-only -verify -Wunused -Wunused-member-function -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wunused -Wunused-member-function -std=c++11 %s

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
  void tf() { }
  template <> void tf<int>() { }  // expected-warning{{unused}}
  
  struct VS {
    virtual void vm() { }
  };
  
  struct SVS : public VS {
    void vm() { }
  };
}

void S::m3() { }  // expected-warning{{unused}}

static inline void f4() { }
const unsigned int cx = 0;

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
  static const double d = 0.0;
  int y = sizeof(d);
}
