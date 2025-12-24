// RUN: %clang_cc1 -std=c11 -fsyntax-only -fdefer-ts -verify %s
// RUN: %clang_cc1 -std=c23 -fsyntax-only -fdefer-ts -verify %s

#define defer _Defer

int g(void);
int h(int x);

void f1(void) {
  defer 1; // expected-warning {{expression result unused}}
  defer 1 + 1; // expected-warning {{expression result unused}}
  defer "a"; // expected-warning {{expression result unused}}
  defer "a" "b" "c"; // expected-warning {{expression result unused}}
  defer defer 1; // expected-warning {{expression result unused}}
  defer defer defer defer 1; // expected-warning {{expression result unused}}
  defer (int) 4; // expected-warning {{expression result unused}}
  defer g();

  defer {}
  defer { defer {} }
  defer { defer {} defer {} }

  defer if (g()) g();
  defer while (g()) g();
  defer for (int i = 0; i < 10; i++) h(i);
  defer switch (g()) { case 1: g(); }

  defer; // expected-warning {{defer statement has empty body}} expected-note {{put the semicolon on a separate line}}
  defer
    ;

  defer a: g(); // expected-error {{substatement of defer must not be a label}}
  defer b: {} // expected-error {{substatement of defer must not be a label}}
  defer { c: g(); }

  if (g()) defer g();
  while (g()) defer g();
  defer ({});
  ({ defer g(); });

  defer int x; // expected-error {{expected expression}}
  defer void q() {} // expected-error {{expected expression}}
}

void f2(void) {
  [[some, attributes]] defer g(); // expected-warning 2 {{unknown attribute}}
  __attribute__((some_attribute)) defer g(); // expected-warning {{unknown attribute}}
  [[some, attributes]] defer { g(); } // expected-warning 2 {{unknown attribute}}
  __attribute__((some_attribute)) defer { g(); } // expected-warning {{unknown attribute}}
}

void f3(void) {
  _Defer 1; // expected-warning {{expression result unused}}
  _Defer {}
  _Defer _Defer {}
  _Defer { defer {} _Defer {} }
  _Defer if (g()) g();
}
