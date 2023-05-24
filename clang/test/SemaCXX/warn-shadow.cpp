// RUN: %clang_cc1 -verify -fsyntax-only -std=c++17 -Wshadow-all %s

namespace {
  int i; // expected-note {{previous declaration is here}}
  static int s; // expected-note {{previous declaration is here}}
}

namespace one {
namespace two {
  int j; // expected-note {{previous declaration is here}}
  typedef int jj; // expected-note 2 {{previous declaration is here}}
  using jjj=int; // expected-note 2 {{previous declaration is here}}
}
}

namespace xx {
  int m;
  typedef int mm;
  using mmm=int;

}
namespace yy {
  int m;
  typedef char mm;
  using mmm=char;
}

using namespace one::two;
using namespace xx;
using namespace yy;

void foo() {
  int i; // expected-warning {{declaration shadows a variable in namespace '(anonymous)'}}
  int j; // expected-warning {{declaration shadows a variable in namespace 'one::two'}}
  static int s; // expected-warning {{declaration shadows a variable in namespace '(anonymous)'}}
  int m;
  int mm;
  int mmm;
}

class A {
  static int data; // expected-note 1 {{previous declaration}}
  // expected-note@+1 1 {{previous declaration}}
  int field;
  int f1, f2, f3, f4; // expected-note 9 {{previous declaration is here}}

  typedef int a1; // expected-note 2 {{previous declaration}}
  using a2=int; // expected-note 2 {{previous declaration}}

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
    char *a1; // no warning 
    char *a2; // no warning
    char *jj; // no warning
    char *jjj; // no warning
    static char *f1; // expected-warning {{declaration shadows a field of 'A'}}
  }

  void test2() {
    typedef char field; // no warning
    typedef char data; // no warning
    typedef char a1; // expected-warning {{declaration shadows a typedef in 'A'}}
    typedef char a2; // expected-warning {{declaration shadows a type alias in 'A'}}
    typedef char jj; // expected-warning {{declaration shadows a typedef in namespace 'one::two'}}
    typedef char jjj; // expected-warning {{declaration shadows a type alias in namespace 'one::two'}}
  }

  void test3() {
    using field=char; // no warning
    using data=char; // no warning
    using a1=char; // expected-warning {{declaration shadows a typedef in 'A'}}
    using a2=char; // expected-warning {{declaration shadows a type alias in 'A'}}
    using jj=char; // expected-warning {{declaration shadows a typedef in namespace 'one::two'}}
    using jjj=char; // expected-warning {{declaration shadows a type alias in namespace 'one::two'}}
  }
};

struct path {
  using value_type = char;
  typedef char value_type2;
  struct iterator {
    using value_type = path; // no warning
    typedef path value_type2; // no warning
  };
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
  static void Baz1();
  static void Baz2();
private:
  int Bar;
};

void Foo::Baz() {
  double Bar = 12; // Don't warn.
}

void Foo::Baz1() {
  typedef int Bar; // Don't warn.
}

void Foo::Baz2() {
  using Bar=int; // Don't warn.
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

extern int bob; // expected-note 1 {{previous declaration is here}}
typedef int bob1; // expected-note 2 {{previous declaration is here}}
using bob2=int; // expected-note 2 {{previous declaration is here}}

// rdar://8883302
void rdar8883302() {
  extern int bob; // don't warn for shadowing.
}

void test8() {
  int bob; // expected-warning {{declaration shadows a variable in the global namespace}}
  int bob1; //no warning
  int bob2; // no warning
}

void test9() {
  typedef int bob; // no warning
  typedef int bob1; // expected-warning {{declaration shadows a typedef in the global namespace}}
  typedef int bob2; // expected-warning {{declaration shadows a type alias in the global namespace}}
}

void test10() {
  using bob=int; // no warning
  using bob1=int; // expected-warning {{declaration shadows a typedef in the global namespace}}
  using bob2=int; // expected-warning {{declaration shadows a type alias in the global namespace}}
}

namespace rdar29067894 {

void avoidWarningWhenRedefining(int b) { // expected-note {{previous definition is here}}
  int a = 0; // expected-note {{previous definition is here}}
  int a = 1; // expected-error {{redefinition of 'a'}}
  int b = 2; // expected-error {{redefinition of 'b'}}

  using c=char; // expected-note {{previous definition is here}}
  using c=int; // expected-error {{type alias redefinition with different types ('int' vs 'char')}}

  typedef char d; // expected-note {{previous definition is here}}
  typedef int d; // expected-error {{typedef redefinition with different types ('int' vs 'char')}}

  using e=char; // expected-note {{previous definition is here}}
  typedef int e; // expected-error {{type alias redefinition with different types ('int' vs 'char')}}

  int f; // expected-note {{previous definition is here}}
  using f=int; // expected-error {{redefinition of 'f'}}

  using g=int; // expected-note {{previous definition is here}}
  int g; // expected-error {{redefinition of 'g'}}

  typedef int h; // expected-note {{previous definition is here}}
  int h; // expected-error {{redefinition of 'h'}}

  int k; // expected-note {{previous definition is here}}
  typedef int k; // expected-error {{redefinition of 'k'}}

  using l=char; // no warning or error.
  using l=char; // no warning or error.
  typedef char l; // no warning or error.
 
  typedef char n; // no warning or error. 
  typedef char n; // no warning or error.
  using n=char; // no warning or error.
}

}

extern "C" {
typedef int externC; // expected-note {{previous declaration is here}}
}
void handleLinkageSpec() {
  typedef void externC; // expected-warning {{declaration shadows a typedef in the global namespace}}
}

namespace PR33947 {
void f(int a) {
  struct A {
    void g(int a) {}
    A() { int a; }
  };
}
}

namespace PR34120 {
struct A {
  int B; // expected-note 2 {{declared here}}
};

class C : public A {
  void D(int B) {} // expected-warning {{parameter 'B' shadows member inherited from type 'A'}}
  void E() {
    extern void f(int B); // Ok
  }
  void F(int B); // Ok, declaration; not definition.
  void G(int B);
};

void C::G(int B) { // expected-warning {{parameter 'B' shadows member inherited from type 'A'}}
}

class Private {
  int B;
};
class Derived : Private {
  void D(int B) {} // Ok
};

struct Static {
  static int B;
};

struct Derived2 : Static {
  void D(int B) {}
};
}

int PR24718;
enum class X { PR24718 }; // Ok, not shadowing

struct PR24718_1;
struct PR24718_2 {
  enum {
    PR24718_1 // Does not shadow a type.
  };
};

namespace structured_binding_tests {
int x; // expected-note {{previous declaration is here}}
int y; // expected-note {{previous declaration is here}}
struct S {
  int a, b;
};

void test1() {
  const auto [x, y] = S(); // expected-warning 2 {{declaration shadows a variable in namespace 'structured_binding_tests'}}
}

void test2() {
  int a; // expected-note {{previous declaration is here}}
  bool b; // expected-note {{previous declaration is here}}
  {
    auto [a, b] = S(); // expected-warning 2 {{declaration shadows a local variable}}
  }
}

class A
{
  int m_a; // expected-note {{previous declaration is here}}
  int m_b; // expected-note {{previous declaration is here}}

  void test3() {
    auto [m_a, m_b] = S(); // expected-warning 2 {{declaration shadows a field of 'structured_binding_tests::A'}}
  }
};

void test4() {
  const auto [a, b] = S(); // expected-note 3 {{previous declaration is here}}
  {
    int a = 4; // expected-warning {{declaration shadows a structured binding}}
  }
  {
    const auto [a, b] = S(); // expected-warning 2 {{declaration shadows a structured binding}}
  }
}

}; // namespace structured_binding_tests
