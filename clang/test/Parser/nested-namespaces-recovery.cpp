// RUN: cp %s %t
// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: not %clang_cc1 -x c++ -fixit %t
// RUN: %clang_cc1 -x c++ %t

namespace foo1::foo2::foo3 { // expected-error {{nested namespace definition must define each namespace separately}}
  int foo(int x) { return x; }
}

int foo(int x) {
  return foo1::foo2::foo3::foo(x);
}

namespace bar1 {
  namespace bar2 {
    namespace bar3 {
      int bar(int x) { return x; }
    }
  }
}

int bar(int x) {
  return bar1::bar2::bar3::bar(x);
}
