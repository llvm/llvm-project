// RUN: %clang_cc1 -std=c23 -fdefer-ts -fsyntax-only -verify -Wredundant-defer %s
// RUN: %clang_cc1 -std=c23 -fdefer-ts -fsyntax-only -fdiagnostics-parseable-fixits -Wredundant-defer %s 2>&1 | FileCheck %s

void f1() {
  if (true) _Defer {} // expected-warning {{redundant use of '_Defer'}} \
                      // expected-note {{did you mean to use '_Defer if'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:3-[[@LINE-2]]:3}:"_Defer "
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:13-[[@LINE-3]]:20}:""

  _Defer {} // expected-warning {{redundant use of '_Defer'}}
}

#define defer _Defer

void f2() {
  defer {} // expected-warning {{redundant use of 'defer'}}
}

void f3() {
  defer {} // OK

  defer defer {} // expected-warning {{redundant use of 'defer'}}

  defer defer defer {} // expected-warning 2 {{redundant use of 'defer'}}

  l1: defer defer {} // expected-warning {{redundant use of 'defer'}}

  l2: defer defer defer {} // expected-warning 2 {{redundant use of 'defer'}}

  l3: l4: defer defer {} // expected-warning {{redundant use of 'defer'}}

  defer [[clang::likely]] defer {} // expected-warning {{redundant use of 'defer'}}

  defer
    [[clang::likely]] [[clang::suppress]]
    defer {}; // expected-warning {{redundant use of 'defer'}}

  [[clang::likely]] defer defer {} // expected-warning {{redundant use of 'defer'}}

  l5:
  [[clang::likely]] defer
    defer {} // expected-warning {{redundant use of 'defer'}}

  defer __attribute__((suppress))
    defer {} // expected-warning {{redundant use of 'defer'}}

  __attribute__((suppress)) defer
    defer {} // expected-warning {{redundant use of 'defer'}}

  l6:
  __attribute__((unknown)) defer // expected-warning {{unknown attribute}}
    defer {} // expected-warning {{redundant use of 'defer'}}

  {
    defer {} // OK
    f1();
  }

  { defer {} } // expected-warning {{redundant use of 'defer'}}

  { defer defer {} } // expected-warning 2 {{redundant use of 'defer'}}

  {
    defer { // OK
      defer {} // OK
      f1();
    }
    f1();
  }

  {
    defer { // expected-warning {{redundant use of 'defer'}}
      defer defer {} // expected-warning 2 {{redundant use of 'defer'}}
    }
  }

  {
    [[clang::likely]] defer { // expected-warning {{redundant use of 'defer'}}
      [[clang::likely]]
      defer defer {} // expected-warning 2 {{redundant use of 'defer'}}
    }
  }

  if (true) {
    defer {} // OK
    f1();
  }

  if (true) defer {} // expected-warning {{redundant use of 'defer'}} \
                     // expected-note {{did you mean to use 'defer if'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:3-[[@LINE-2]]:3}:"defer "
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:13-[[@LINE-3]]:19}:""

  if (true)
    defer {} // expected-warning {{redundant use of 'defer'}} \
             // expected-note {{did you mean to use 'defer if'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:3-[[@LINE-3]]:3}:"defer "
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:5-[[@LINE-3]]:11}:""

  if (true) { defer {} } // expected-warning {{redundant use of 'defer'}} \
                         // expected-note {{did you mean to use 'defer if'}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:3-[[@LINE-2]]:3}:"defer "
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:15-[[@LINE-3]]:21}:""

  if (true) {
    defer {} // expected-warning {{redundant use of 'defer'}} \
             // expected-note {{did you mean to use 'defer if'}}
  }
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:3-[[@LINE-4]]:3}:"defer "
  // CHECK: fix-it:"{{.*}}":{[[@LINE-4]]:5-[[@LINE-4]]:11}:""

  for (;;) {
    defer {} // OK
    f1();
  }

  for (;;)
    defer {} // expected-warning {{redundant use of 'defer'}}

  for (;;) {
    defer {} // expected-warning {{redundant use of 'defer'}}
  }

  for (;;) {
    defer {} // OK
    f1();
    break;
  }

  for (;;) {
    defer {} // expected-warning {{redundant use of 'defer'}}
    break;
  }

  for (;;) {
    defer {} // OK
    f1();
    l7: break;
  }

  for (;;) {
    defer {} // expected-warning {{redundant use of 'defer'}}
    l8: break;
  }

  for (;;) {
    defer {} // expected-warning {{redundant use of 'defer'}}
    l9: l10: break;
  }

  for (;;) {
    defer {} // OK
    f1();
    [[clang::likely]] break;
  }

  for (;;) {
    defer {} // expected-warning {{redundant use of 'defer'}}
    [[clang::likely]] break;
  }

  for (;;) {
    defer {} // expected-warning {{redundant use of 'defer'}}
    [[clang::likely]] [[clang::suppress]] break;
  }

  for (;;) {
    defer {} // expected-warning {{redundant use of 'defer'}}
    __attribute__((suppress)) continue;
  }

  for (;;) {
    defer {} // expected-warning {{redundant use of 'defer'}}
    l11: [[clang::likely]] break;
  }

  for (;;) {
    defer {} // OK
    f1();
    continue;
  }

  for (;;) {
    defer {} // expected-warning {{redundant use of 'defer'}}
    continue;
  }

  for (;;) {
    defer {} // OK
    f1();
    l12: continue;
  }

  for (;;) {
    defer {} // expected-warning {{redundant use of 'defer'}}
    l13: continue;
  }

  for (;;) {
    defer {} // expected-warning {{redundant use of 'defer'}}
    l14: l15: continue;
  }

  for (;;) {
    defer {} // OK
    f1();
    [[clang::likely]] continue;
  }

  for (;;) {
    defer {} // expected-warning {{redundant use of 'defer'}}
    [[clang::likely]] continue;
  }

  for (;;) {
    defer {} // expected-warning {{redundant use of 'defer'}}
    [[clang::likely]] [[clang::suppress]] continue;
  }

  for (;;) {
    defer {} // expected-warning {{redundant use of 'defer'}}
    __attribute__((suppress)) continue;
  }

  for (;;) {
    defer {} // expected-warning {{redundant use of 'defer'}}
    l16: [[clang::likely]] continue;
  }

  while (true) {
    defer {} // OK
    f1();
  }

  while (true)
    defer {} // expected-warning {{redundant use of 'defer'}}

  while (true) {
    defer {} // expected-warning {{redundant use of 'defer'}}
  }

  while (true) {
    defer {} // OK
    f1();
    break;
  }

  while (true) {
    defer {} // expected-warning {{redundant use of 'defer'}}
    break;
  }

  while (true) {
    defer {} // OK
    f1();
    continue;
  }

  while (true) {
    defer {} // expected-warning {{redundant use of 'defer'}}
    continue;
  }

  do {
    defer {} // OK
    f1();
  } while (true);

  do {
    defer {} // expected-warning {{redundant use of 'defer'}}
  } while (true);

  do {
    defer {} // OK
    f1();
    break;
  } while (true);

  do {
    defer {} // expected-warning {{redundant use of 'defer'}}
    break;
  } while (true);

  do {
    defer {} // OK
    f1();
    continue;
  } while (true);

  do {
    defer {} // expected-warning {{redundant use of 'defer'}}
    continue;
  } while (true);

  defer {} // expected-warning {{redundant use of 'defer'}}
}

int f4() {
  defer {} // OK
  return 0;
}

void f5() {
  defer {} // OK
  f1();
  return;
}

void f6() {
  defer {} // expected-warning {{redundant use of 'defer'}}
  return;
}

void f7() {
  defer {} // expected-warning {{redundant use of 'defer'}}
  l17: return;
}

void f8() {
  defer {} // expected-warning {{redundant use of 'defer'}}
  [[clang::likely]] return;
}

void f9() {
  defer {} // expected-warning {{redundant use of 'defer'}}
  l18: l19: [[clang::likely]] [[clang::suppress]] return;
}
