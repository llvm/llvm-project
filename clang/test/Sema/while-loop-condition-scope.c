// RUN: %clang_cc1 -fsyntax-only -verify %s 

void f() {
  while (({ continue; 1; })) {
    // expected-error@-1 {{'continue' statement not in loop statement}}

  }
  while (({ break; 1; })) {
    // expected-error@-1 {{'break' statement not in loop or switch statement}}
  }
}
