// RUN: %clang_cc1 -std=c++11 -verify=expected,expected11,both,both11 %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -std=c++14 -verify=expected,expected14,both        %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -std=c++11 -verify=ref,ref11,both,both11           %s
// RUN: %clang_cc1 -std=c++14 -verify=ref,ref14,both                  %s

namespace Simple {
  struct S {
    mutable int a; // both-note {{declared here}} \
                   // both11-note {{declared here}}
    int a2;
  };

  constexpr S s{12, 24};
  static_assert(s.a == 12, ""); // both-error {{not an integral constant expression}}  \
                                // both-note {{read of mutable member 'a'}}
  static_assert(s.a2 == 24, "");


  constexpr S s2{12, s2.a}; // both11-error {{must be initialized by a constant expression}} \
                            // both11-note {{read of mutable member 'a'}} \
                            // both11-note {{declared here}}
  static_assert(s2.a2 == 12, ""); // both11-error {{not an integral constant expression}} \
                                  // both11-note {{initializer of 's2' is not a constant expression}}
}
#if __cplusplus >= 201402L
namespace ConstInMutable {
  class B {
    public:

    const int f;
    constexpr B() : f(12) {}
  };
  class A {
    public:
    mutable B b;
    constexpr A() = default;
  };
  constexpr int constInMutable() {
    A a;

    int *m = (int*)&a.b.f;
    *m = 12; // both-note {{modification of object of const-qualified type 'const int' is not allowed in a constant expression}}
    return 1;
  }
  static_assert(constInMutable() == 1, ""); // both-error {{not an integral constant expression}} \
                                            // both-note {{in call to}}
}

namespace MutableInConst {
  class C {
  public:
    mutable int c;
    constexpr C() : c(50) {}
  };
  class D {
  public:
    C c;
    constexpr D() {}
  };
  constexpr int mutableInConst() {
    const D d{};
    int *m = (int*)&d.c.c;
    *m = 12;
    return 1;
  }
  static_assert(mutableInConst() == 1, "");
}
#endif

struct D { mutable int y; }; // both-note {{declared here}}
constexpr D d1 = { 1 };
constexpr D d2 = d1; // both-error {{must be initialized by a constant expression}} \
                     // both-note {{read of mutable member 'y}} \
                     // both-note {{in call to}}


