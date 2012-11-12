// RUN: %clang_cc1 -analyze -analyzer-checker=core,debug.ExprInspection %s -verify

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