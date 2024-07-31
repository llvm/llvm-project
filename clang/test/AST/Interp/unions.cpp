// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -verify=ref,both %s

union U {
  int a;
  int b;
};

constexpr U a = {12};
static_assert(a.a == 12, "");
static_assert(a.b == 0, ""); // both-error {{not an integral constant expression}} \
                             // both-note {{read of member 'b' of union with active member 'a'}}
union U1 {
  int i;
  float f = 3.0f;
};
constexpr U1 u1{};
static_assert(u1.f == 3.0, "");
static_assert(u1.i == 1, ""); // both-error {{not an integral constant expression}} \
                              // both-note {{read of member 'i' of union with active member 'f'}}



union A {
  int a;
  double d;
};
constexpr A aa = {1, 2.0}; // both-error {{excess elements in union initializer}}
constexpr A ab = {.d = 1.0};
static_assert(ab.d == 1.0, "");
static_assert(ab.a == 1, ""); // both-error {{not an integral constant expression}} \
                              // both-note {{read of member 'a' of union with active member 'd'}}


namespace Empty {
  union E {};
  constexpr E e{};
}

namespace SimpleStore {
  union A {
    int a;
    int b;
  };
  constexpr int foo() {
    A a{.b = 4};
    a.b = 10;
    return a.b;
  }
  static_assert(foo() == 10, "");

  constexpr int empty() {
    A a{}; /// Just test that this works.
    return 10;
  }
  static_assert(empty() == 10, "");
}

namespace ZeroInit {
  struct S { int m; };
  union Z {
    float f;
  };

  constexpr Z z{};
  static_assert(z.f == 0.0, "");
}
