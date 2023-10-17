// RUN: %clang_cc1 -fsyntax-only -Wunused-variable -Wunused-label -Wno-c++1y-extensions -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wunused-variable -Wunused-label -Wno-c++1y-extensions -verify -std=c++11 %s
template<typename T> void f() {
  T t;
  t = 17;
}

// PR5407
struct A { A(); };
struct B { ~B(); };
void f() {
  A a;
  B b;
}

// PR5531
namespace PR5531 {
  struct A {
  };

  struct B {
    B(int);
  };

  struct C {
    ~C();
  };

  void test() {
    A();
    B(17);
    C();
  }
}

template<typename T>
struct X0 { };

template<typename T>
void test_dependent_init(T *p) {
  X0<int> i(p);
  (void)i;
}

void unused_local_static() {
  static int x = 0;
  static int y = 0; // expected-warning{{unused variable 'y'}}
#pragma unused(x)
  static __attribute__((used)) int z;
  static __attribute__((unused)) int w;
  [[maybe_unused]] static int v;
}

// PR10168
namespace PR10168 {
  // We expect a warning in the definition only for non-dependent variables, and
  // a warning in the instantiation only for dependent variables.
  template<typename T>
  struct S {
    void f() {
      int a; // expected-warning {{unused variable 'a'}}
      T b; // expected-warning 2{{unused variable 'b'}}
    }
  };

  template<typename T>
  void f() {
    int a; // expected-warning {{unused variable 'a'}}
    T b; // expected-warning 2{{unused variable 'b'}}
  }

  void g() {
    S<int>().f(); // expected-note {{here}}
    S<char>().f(); // expected-note {{here}}
    f<int>(); // expected-note {{here}}
    f<char>(); // expected-note {{here}}
  }
}

namespace PR11550 {
  struct S1 {
    S1();
  };
  S1 makeS1();
  void testS1(S1 a) {
    // This constructor call can be elided.
    S1 x = makeS1(); // expected-warning {{unused variable 'x'}}

    // This one cannot, so no warning.
    S1 y;

    // This call cannot, but the constructor is trivial.
    S1 z = a; // expected-warning {{unused variable 'z'}}
  }

  // The same is true even when we know thet constructor has side effects.
  void foo();
  struct S2 {
    S2() {
      foo();
    }
  };
  S2 makeS2();
  void testS2(S2 a) {
    S2 x = makeS2(); // expected-warning {{unused variable 'x'}}
    S2 y;
    S2 z = a; // expected-warning {{unused variable 'z'}}
  }

  // Or when the constructor is not declared by the user.
  struct S3 {
    S1 m;
  };
  S3 makeS3();
  void testS3(S3 a) {
    S3 x = makeS3(); // expected-warning {{unused variable 'x'}}
    S3 y;
    S3 z = a; // expected-warning {{unused variable 'z'}}
  }
}

namespace PR19305 {
  template<typename T> int n = 0; // no warning
  int a = n<int>;

  template<typename T> const int l = 0; // no warning
  int b = l<int>;

  // PR19558
  template<typename T> const int o = 0; // no warning
  template<typename T> const int o<T*> = 0; // no warning
  int c = o<int*>;

  template<> int o<void> = 0; // no warning
  int d = o<void>;

  // FIXME: It'd be nice to warn here.
  template<typename T> int m = 0;
  template<typename T> int m<T*> = 0;

  // This has external linkage, so could be referenced by a declaration in a
  // different translation unit.
  template<> const int m<void> = 0; // no warning
}

namespace ctor_with_cleanups {
  struct S1 {
    ~S1();
  };
  struct S2 {
    S2(const S1&);
  };
  void func() {
    S2 s((S1()));
  }
}

#include "Inputs/warn-unused-variables.h"

class NonTriviallyDestructible {
public:
  ~NonTriviallyDestructible() {}
};

namespace arrayRecords {

struct Foo {
  int x;
  Foo(int x) : x(x) {}
};

struct Elidable {
  Elidable();
};

void foo(int size) {
  Elidable elidable; // no warning
  Elidable elidableArray[2]; // no warning
  Elidable elidableDynArray[size]; // no warning
  Elidable elidableNestedArray[1][2][3]; // no warning

  NonTriviallyDestructible scalar; // no warning
  NonTriviallyDestructible array[2];  // no warning
  NonTriviallyDestructible nestedArray[2][2]; // no warning

  Foo fooScalar = 1; // expected-warning {{unused variable 'fooScalar'}}
  Foo fooArray[] = {1,2}; // expected-warning {{unused variable 'fooArray'}}
  Foo fooNested[2][2] = { {1,2}, {3,4} }; // expected-warning {{unused variable 'fooNested'}}
}

template<int N>
void bar() {
  NonTriviallyDestructible scaler; // no warning
  NonTriviallyDestructible array[N]; // no warning
}

void test() {
  foo(10);
  bar<2>();
}

} // namespace arrayRecords

#if __cplusplus >= 201103L
namespace with_constexpr {
template <typename T>
struct Literal {
  T i;
  Literal() = default;
  constexpr Literal(T i) : i(i) {}
};

struct NoLiteral {
  int i;
  NoLiteral() = default;
  constexpr NoLiteral(int i) : i(i) {}
  ~NoLiteral() {}
};

static Literal<int> gl1;          // expected-warning {{unused variable 'gl1'}}
static Literal<int> gl2(1);       // expected-warning {{unused variable 'gl2'}}
static const Literal<int> gl3(0); // expected-warning {{unused variable 'gl3'}}

template <typename T>
void test(int i) {
  Literal<int> l1;     // expected-warning {{unused variable 'l1'}}
  Literal<int> l2(42); // expected-warning {{unused variable 'l2'}}
  Literal<int> l3(i);  // no-warning
  Literal<T> l4(0);    // no-warning
  NoLiteral nl1;       // no-warning
  NoLiteral nl2(42);   // no-warning
}
}

namespace crash {
struct a {
  a(const char *);
};
template <typename b>
void c() {
  a d(b::e ? "" : "");
}
}

// Ensure we don't warn on dependent constructor calls.
namespace dependent_ctor {
struct S {
  S() = default;
  S(const S &) = default;
  S(int);
};

template <typename T>
void foo(T &t) {
  S s{t};
}
}
#endif

// Ensure we do not warn on lifetime extension
namespace gh54489 {

void f() {
  const auto &a = NonTriviallyDestructible();
  const auto &b = a; // expected-warning {{unused variable 'b'}}
#if __cplusplus >= 201103L
  const auto &&c = NonTriviallyDestructible();
  auto &&d = c; // expected-warning {{unused variable 'd'}}
#endif
}

struct S {
  S() = default;
  S(const S &) = default;
  S(int);
};

template <typename T>
void foo(T &t) {
  const auto &extended = S{t};
}

void test_foo() {
  int i;
  foo(i);
}

struct RAIIWrapper {
  RAIIWrapper();
  ~RAIIWrapper();
};

void RAIIWrapperTest() {
  auto const guard = RAIIWrapper();
  auto const &guard2 = RAIIWrapper();
  auto &&guard3 = RAIIWrapper();
}

} // namespace gh54489
