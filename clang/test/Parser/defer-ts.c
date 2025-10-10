// RUN: %clang_cc1 -std=c11 -fsyntax-only -fdefer-ts -verify %s
// RUN: %clang_cc1 -std=c23 -fsyntax-only -fdefer-ts -verify %s

int g(void);
int h(int x);

void f(void) {
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

  defer a: g(); // expected-error {{body of 'defer' statement cannot start with a label}}
  defer b: {} // expected-error {{body of 'defer' statement cannot start with a label}}
  defer { c: g(); }

  if (g()) defer g();
  while (g()) defer g();
  defer ({});
  ({ defer g(); });

  defer int x; // expected-error {{expected expression}}
  defer void q() {} // expected-error {{expected expression}}
}
