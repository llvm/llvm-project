// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection %s -verify

void clang_analyzer_checkInlined(bool);
void clang_analyzer_eval(int);

namespace EnumsViaMemberExpr {
  struct Foo {
    enum E {
      Bar = 1
    };
  };

  void testEnumVal(Foo Baz) {
    clang_analyzer_eval(Baz.Bar == Foo::Bar); // expected-warning{{TRUE}}
  }

  void testEnumRef(Foo &Baz) {
    clang_analyzer_eval(Baz.Bar == Foo::Bar); // expected-warning{{TRUE}}
  }

  void testEnumPtr(Foo *Baz) {
    clang_analyzer_eval(Baz->Bar == Foo::Bar); // expected-warning{{TRUE}}
  }
}

namespace PR19531 {
  struct A {
    A() : x(0) {}
    bool h() const;
    int x;
  };

  struct B {
    void g(bool (A::*mp_f)() const) {
      // This used to trigger an assertion because the 'this' pointer is a
      // temporary.
      (A().*mp_f)();
    }
    void f() { g(&A::h); }
  };
}
