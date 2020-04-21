// RUN: %clang_cc1 -std=c++11 -fsyntax-only -frecovery-ast -verify %s

// NOTE: these tests can be merged to existing tests after -frecovery-ast is
// turned on by default.
void test1() {
  struct Data {};
  struct T {
    Data *begin();
    Data *end();
  };
  T *pt;
  for (Data *p : T()) {} // expected-error {{no viable conversion from 'Data' to 'Data *'}}
                         // expected-note@-5 {{selected 'begin' function with iterator type}}
}

void test2() {
  struct Bottom {
    constexpr Bottom() {}
  };
  struct Base : Bottom {
    constexpr Base(int a = 42, const char *b = "test") : a(a), b(b) {}
    int a;
    const char *b;
  };
  constexpr Base *nullB = 12; // expected-error {{cannot initialize a variable of type}}
  // verify that the "static_assert expression is not an integral constant expr"
  // diagnostic is suppressed.
  static_assert((Bottom*)nullB == 0, "");
}
