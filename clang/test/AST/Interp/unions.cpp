// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -std=c++20 -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -verify=ref,both %s
// RUN: %clang_cc1 -std=c++20 -verify=ref,both %s

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

namespace DefaultInit {
  union U1 {
    constexpr U1() {}
    int a, b = 42;
  };

  constexpr U1 u1; /// OK.

  constexpr int foo() {
    U1 u;
    return u.a; // both-note {{read of member 'a' of union with active member 'b'}}
  }
  static_assert(foo() == 42); // both-error {{not an integral constant expression}} \
                              // both-note {{in call to}}
}

#if __cplusplus >= 202002L
namespace SimpleActivate {
  constexpr int foo() { // ref-error {{never produces a constant expression}}
    union {
      int a;
      int b;
    } Z;

    Z.a = 10;
    Z.b = 20;
    return Z.a; // both-note {{read of member 'a' of union with active member 'b'}} \
                // ref-note {{read of member 'a' of union with active member 'b}}
  }
  static_assert(foo() == 20); // both-error {{not an integral constant expression}} \
                              // both-note {{in call to}}

  constexpr int foo2() {
    union {
      int a;
      int b;
    } Z;

    Z.a = 10;
    Z.b = 20;
    return Z.b;
  }
  static_assert(foo2() == 20);


  constexpr int foo3() {
    union {
      struct {
        float x,y;
      } a;
      int b;
    } Z;

    Z.a.y = 10;

    return Z.a.x; // both-note {{read of uninitialized object}}
  }
  static_assert(foo3() == 10); // both-error {{not an integral constant expression}} \
                               // both-note {{in call to}}

  constexpr int foo4() {
    union {
      struct {
        float x,y;
      } a;
      int b;
    } Z;

    Z.a.x = 100;
    Z.a.y = 10;

    return Z.a.x;
  }
  static_assert(foo4() == 100);
}

namespace IndirectFieldDecl {
  struct C {
    union { int a, b = 2, c; };
    union { int d, e = 5, f; };
    constexpr C() : a(1) {}
  };
  static_assert(C().a == 1, "");
}

namespace UnionDtor {

  union U {
    int *I;
    constexpr U(int *I) : I(I) {}
    constexpr ~U() {
      *I = 10;
    }
  };

  constexpr int foo() {
    int a = 100;
    {
      U u(&a);
    }
    return a;
  }
  static_assert(foo() == 10);
}

namespace UnionMemberDtor {
  class UM {
  public:
    int &I;
    constexpr UM(int &I) : I(I) {}
    constexpr ~UM() { I = 200; }
  };

  union U {
    UM um;
    constexpr U(int &I) : um(I) {}
    constexpr ~U() {
    }
  };

  constexpr int foo() {
    int a = 100;
    {
      U u(a);
    }

    return a;
  }
  static_assert(foo() == 100);
}

namespace Nested {
  union U {
    int a;
    int b;
  };

  union U2 {
    U u;
    U u2;
    int x;
    int y;
  };

 constexpr int foo() { // ref-error {{constexpr function never produces a constant expression}}
    U2 u;
    u.u.a = 10;
    int a = u.y; // both-note {{read of member 'y' of union with active member 'u' is not allowed in a constant expression}} \
                 // ref-note {{read of member 'y' of union with active member 'u' is not allowed in a constant expression}}

    return 1;
  }
  static_assert(foo() == 1); // both-error {{not an integral constant expression}} \
                             // both-note {{in call to}}

 constexpr int foo2() {
    U2 u;
    u.u.a = 10;
    return u.u.a;
  }
  static_assert(foo2() == 10);

 constexpr int foo3() { // ref-error {{constexpr function never produces a constant expression}}
    U2 u;
    u.u.a = 10;
    int a = u.u.b; // both-note {{read of member 'b' of union with active member 'a' is not allowed in a constant expression}} \
                   // ref-note {{read of member 'b' of union with active member 'a' is not allowed in a constant expression}}

    return 1;
  }
  static_assert(foo3() == 1); // both-error {{not an integral constant expression}} \
                              // both-note {{in call to}}

  constexpr int foo4() { // ref-error {{constexpr function never produces a constant expression}}
    U2 u;

    u.x = 10;

    return u.u.a;// both-note {{read of member 'u' of union with active member 'x' is not allowed in a constant expression}} \
                 // ref-note {{read of member 'u' of union with active member 'x' is not allowed in a constant expression}}
  }
  static_assert(foo4() == 1); // both-error {{not an integral constant expression}} \
                              // both-note {{in call to}}

}
#endif
