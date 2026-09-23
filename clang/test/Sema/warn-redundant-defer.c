// RUN: %clang_cc1 -std=c23 -fdefer-ts -fsyntax-only -verify %s -Wredundant-defer

#define defer _Defer

void f1() {
  defer {} // expected-warning {{redundant use of defer}}
}

void f2() {
  defer {} // OK

  defer defer {} // expected-warning {{redundant use of defer}}

  defer defer defer {} // expected-warning 2 {{redundant use of defer}}

  {
    defer {} // OK
    f1();
  }

  { defer {} } // expected-warning {{redundant use of defer}}

  { defer defer {} } // expected-warning 2 {{redundant use of defer}}

  {
    defer { // OK
      defer {} // OK
      f1();
    }
    f1();
  }

  {
    defer { // expected-warning {{redundant use of defer}}
      defer defer {} // expected-warning 2 {{redundant use of defer}}
    }
  }

  if (true) {
    defer {} // OK
    f1();
  }

  if (true)
    defer {} // expected-warning {{redundant use of defer}}

  if (true) {
    defer {} // expected-warning {{redundant use of defer}}
  }

  for (;;) {
    defer {} // OK
    f1();
  }

  for (;;)
    defer {} // expected-warning {{redundant use of defer}}

  for (;;) {
    defer {} // expected-warning {{redundant use of defer}}
  }

  for (;;) {
    defer {} // OK
    f1();
    break;
  }

  for (;;) {
    defer {} // expected-warning {{redundant use of defer}}
    break;
  }

  for (;;) {
    defer {} // OK
    f1();
    l1: break;
  }

  for (;;) {
    defer {} // expected-warning {{redundant use of defer}}
    l2: break;
  }

  for (;;) {
    defer {} // expected-warning {{redundant use of defer}}
    l3: l4: break;
  }

  for (;;) {
    defer {} // OK
    f1();
    [[clang::likely]] break;
  }

  for (;;) {
    defer {} // expected-warning {{redundant use of defer}}
    [[clang::likely]] break;
  }

  for (;;) {
    defer {} // expected-warning {{redundant use of defer}}
    [[clang::likely]] [[clang::suppress]] break;
  }

  for (;;) {
    defer {} // OK
    f1();
    continue;
  }

  for (;;) {
    defer {} // expected-warning {{redundant use of defer}}
    continue;
  }

  for (;;) {
    defer {} // OK
    f1();
    l5: continue;
  }

  for (;;) {
    defer {} // expected-warning {{redundant use of defer}}
    l6: continue;
  }

  for (;;) {
    defer {} // expected-warning {{redundant use of defer}}
    l7: l8: continue;
  }

  for (;;) {
    defer {} // OK
    f1();
    [[clang::likely]] continue;
  }

  for (;;) {
    defer {} // expected-warning {{redundant use of defer}}
    [[clang::likely]] continue;
  }

  for (;;) {
    defer {} // expected-warning {{redundant use of defer}}
    [[clang::likely]] [[clang::suppress]] continue;
  }

  while (true) {
    defer {} // OK
    f1();
  }

  while (true)
    defer {} // expected-warning {{redundant use of defer}}

  while (true) {
    defer {} // expected-warning {{redundant use of defer}}
  }

  while (true) {
    defer {} // OK
    f1();
    break;
  }

  while (true) {
    defer {} // expected-warning {{redundant use of defer}}
    break;
  }

  while (true) {
    defer {} // OK
    f1();
    continue;
  }

  while (true) {
    defer {} // expected-warning {{redundant use of defer}}
    continue;
  }

  do {
    defer {} // OK
    f1();
  } while (true);

  do {
    defer {} // expected-warning {{redundant use of defer}}
  } while (true);

  do {
    defer {} // OK
    f1();
    break;
  } while (true);

  do {
    defer {} // expected-warning {{redundant use of defer}}
    break;
  } while (true);

  do {
    defer {} // OK
    f1();
    continue;
  } while (true);

  do {
    defer {} // expected-warning {{redundant use of defer}}
    continue;
  } while (true);

  defer {} // expected-warning {{redundant use of defer}}
}

int f3() {
  defer {} // OK
  return 0;
}

void f4() {
  defer {} // OK
  f1();
  return;
}

void f5() {
  defer {} // expected-warning {{redundant use of defer}}
  return;
}
