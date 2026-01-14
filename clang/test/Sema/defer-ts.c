// RUN: %clang_cc1 -std=c23 -fdefer-ts -fsyntax-only -verify %s

#define defer _Defer

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
  defer { // expected-note {{jump enters a defer statement}}
    l2:
  }
  goto l2; // expected-error {{cannot jump from this goto statement to its label}}
}

void f6() {
  goto b; // expected-error {{cannot jump from this goto statement to its label}}
  {
    defer {} // expected-note {{jump bypasses defer statement}}
    b:
  }

  {
    defer {} // expected-note {{jump bypasses defer statement}}
    b2:
  }
  goto b2; // expected-error {{cannot jump from this goto statement to its label}}
}

void f7() {
  defer { // expected-note {{jump bypasses defer statement}}
    goto cross1; // expected-error {{cannot jump from this goto statement to its label}}
    cross2:
  }
  defer { // expected-note {{jump exits a defer statement}} expected-note {{jump enters a defer statement}}
    goto cross2; // expected-error {{cannot jump from this goto statement to its label}}
    cross1:
  }
}

void f8() {
  defer {
    return; // expected-error {{cannot return from a defer statement}}
  }

  {
    defer {
      return; // expected-error {{cannot return from a defer statement}}
    }
  }

  switch (1) {
    case 1: defer {
      break; // expected-error {{cannot break out of a defer statement}}
    }
  }

  for (;;) {
    defer {
      break; // expected-error {{cannot break out of a defer statement}}
    }
  }

  for (;;) {
    defer {
      continue; // expected-error {{cannot continue loop outside of enclosing defer statement}}
    }
  }

  switch (1) {
    defer {} // expected-note {{jump bypasses defer statement}}
  default: // expected-error {{cannot jump from switch statement to this case label}}
    defer {}
    break;
  }

  switch (1) {
    case 1: {
      defer { // expected-note {{jump enters a defer statement}}
        case 2: {} // expected-error {{cannot jump from switch statement to this case label}}
      }
    }
  }

  switch (1) {
    case 1: defer {
      switch (2) { case 2: break; }
    }
  }

  for (;;) {
    defer { for (;;) break; }
  }

  for (;;) {
    defer { for (;;) continue; }
  }
}

void f9() {
  {
    defer {}
    goto l1;
  }
  l1:

  {
    goto l2;
    defer {}
  }
  l2:

  {
    { defer {} }
    goto l3;
  }
  l3:

  {
    defer {}
    { goto l4; }
  }
  l4:
}

void f10(int i) {
  switch (i) {
    defer case 12: break; // expected-error {{cannot break out of a defer statement}} \
                             expected-error {{cannot jump from switch statement to this case label}} \
                             expected-note {{jump enters a defer statement}} \
                             expected-note {{jump bypasses defer statement}}

    defer default: break; // expected-error {{cannot break out of a defer statement}} \
                             expected-error {{cannot jump from switch statement to this case label}} \
                             expected-note {{jump enters a defer statement}}
  }
}
