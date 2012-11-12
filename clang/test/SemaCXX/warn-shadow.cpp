// RUN: %clang_cc1 -verify -fsyntax-only -Wshadow %s

namespace {
  int i; // expected-note {{previous declaration is here}}
}

namespace one {
namespace two {
  int j; // expected-note {{previous declaration is here}}
}
}

namespace xx {
  int m;
}
namespace yy {
  int m;
}

using namespace one::two;
using namespace xx;
using namespace yy;

void foo() {
  int i; // expected-warning {{declaration shadows a variable in namespace '<anonymous>'}}
  int j; // expected-warning {{declaration shadows a variable in namespace 'one::two'}}
  int m;
}

class A {
  static int data; // expected-note {{previous declaration}}
  int field; // expected-note {{previous declaration}}

  void test() {
    char *field; // expected-warning {{declaration shadows a field of 'A'}}
    char *data; // expected-warning {{declaration shadows a static data member of 'A'}}
  }
};

// TODO: this should warn, <rdar://problem/5018057>
class B : A {
  int data;
  static int field;
};

// rdar://8900456
namespace rdar8900456 {
struct Foo {
  static void Baz();
private:
  int Bar;
};

void Foo::Baz() {
  double Bar = 12; // Don't warn.
}
}

// http://llvm.org/PR9160
namespace PR9160 {
struct V {
  V(int);
};
struct S {
  V v;
  static void m() {
    if (1) {
      V v(0);
    }
  }
};
}

extern int bob; // expected-note {{previous declaration is here}}

// rdar://8883302
void rdar8883302() {
  extern int bob; // don't warn for shadowing.
}

void test8() {
  int bob; // expected-warning {{declaration shadows a variable in the global namespace}}
}
