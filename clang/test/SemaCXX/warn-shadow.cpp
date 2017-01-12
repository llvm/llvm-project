// RUN: %clang_cc1 -verify -fsyntax-only -Wshadow-all %s

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
  int i; // expected-warning {{declaration shadows a variable in namespace '(anonymous)'}}
  int j; // expected-warning {{declaration shadows a variable in namespace 'one::two'}}
  int m;
}

class A {
  static int data; // expected-note {{previous declaration}}
  // expected-note@+1 {{previous declaration}}
  int field;
  int f1, f2, f3, f4; // expected-note 8 {{previous declaration is here}}

  // The initialization is safe, but the modifications are not.
  A(int f1, int f2, int f3, int f4) // expected-note-re 4 {{variable 'f{{[0-4]}}' is declared here}}
	  : f1(f1) {
    f1 = 3; // expected-warning {{modifying constructor parameter 'f1' that shadows a field of 'A'}}
    f1 = 4; // one warning per shadow
    f2++; // expected-warning {{modifying constructor parameter 'f2' that shadows a field of 'A'}}
    --f3; // expected-warning {{modifying constructor parameter 'f3' that shadows a field of 'A'}}
    f4 += 2; // expected-warning {{modifying constructor parameter 'f4' that shadows a field of 'A'}}
  }

  // The initialization is safe, but the modifications are not.
  // expected-warning-re@+1 4 {{constructor parameter 'f{{[0-4]}}' shadows the field 'f{{[0-9]}}' of 'A'}}
  A(int f1, int f2, int f3, int f4, double overload_dummy) {}

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

namespace rdar29067894 {

void avoidWarningWhenRedefining(int b) { // expected-note {{previous definition is here}}
  int a = 0; // expected-note {{previous definition is here}}
  int a = 1; // expected-error {{redefinition of 'a'}}
  int b = 2; // expected-error {{redefinition of 'b'}}
}

}
