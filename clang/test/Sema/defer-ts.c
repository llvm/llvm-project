// RUN: %clang_cc1 -std=c23 -fdefer-ts -fsyntax-only -verify %s

void a();

void f1() {
  defer {
    goto l1;
    l1:
  }

  defer {
    l2:
    goto l2;
  }
}

void f2() {
  goto l1; // expected-error {{cannot jump from this goto statement to its label}}
  defer { // expected-note {{jump enters a defer statement}}
    l1:
  }

  goto l2; // expected-error {{cannot jump from this goto statement to its label}}
  defer {} // expected-note {{jump bypasses defer statement}}
  l2:
}

void f3() {
  x:
  defer { // expected-note {{jump exits a defer statement}}
    goto x; // expected-error {{cannot jump from this goto statement to its label}}
  }
}

void f4() {
  defer { // expected-note {{jump exits a defer statement}}
    goto y; // expected-error {{cannot jump from this goto statement to its label}}
  }
  y:
}

void f5() {
  defer { // expected-note {{jump bypasses defer statement}}
    goto cross1; // expected-error {{cannot jump from this goto statement to its label}}
    cross2:
  }
  defer { // expected-note {{jump exits a defer statement}} expected-note {{jump enters a defer statement}}
    goto cross2; // expected-error {{cannot jump from this goto statement to its label}}
    cross1:
  }
}
