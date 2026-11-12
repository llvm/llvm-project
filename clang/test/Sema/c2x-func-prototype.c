// RUN: %clang_cc1 -fsyntax-only -verify=c2x -std=c2x %s
// RUN: %clang_cc1 -Wno-strict-prototypes -fsyntax-only -verify -std=c17 %s
// RUN: %clang_cc1 -fsyntax-only -fno-knr-functions -std=c99 -verify=c2x %s
// expected-no-diagnostics

void func(); // c2x-note {{'func' declared here}}
typedef void (*fp)();

void other_func(int i);

void call(void) {
  func(1, 2, 3); // c2x-error {{too many arguments to function call, expected 0, have 3}}
  fp call_me = func;
  call_me(1, 2, 3); // c2x-error {{too many arguments to function call, expected 0, have 3}}

  fp nope = other_func; // c2x-error {{incompatible function pointer types initializing 'fp' (aka 'void (*)(void)') with an expression of type 'void (int)'}}
}

// Ensure these function declarations do not merge in C23.
void redecl1();      // c2x-note {{previous declaration is here}}
void redecl1(int i); // c2x-error {{conflicting types for 'redecl1'}}

void redecl2(int i); // c2x-note {{previous declaration is here}}
void redecl2();      // c2x-error {{conflicting types for 'redecl2'}}
