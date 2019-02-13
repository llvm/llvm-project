// RUN: %clang_cc1 -std=c++1z -verify %s
// RUN: %clang_cc1 -std=c++17 -verify %s

void testIf() {
  int x = 0;
  if (x; x) ++x;
  if (int t = 0; t) ++t; else --t;

  if (int x, y = 0; y) // expected-note 2 {{previous definition is here}}
    int x = 0; // expected-error {{redefinition of 'x'}}
  else
    int x = 0; // expected-error {{redefinition of 'x'}}

  if (x; int a = 0) ++a;
  if (x, +x; int a = 0) // expected-note 2 {{previous definition is here}} expected-warning {{unused}}
    int a = 0; // expected-error {{redefinition of 'a'}}
  else
    int a = 0; // expected-error {{redefinition of 'a'}}

  if (int b = 0; b)
    ;
  b = 2; // expected-error {{use of undeclared identifier}}
}

void testSwitch() {
  int x = 0;
  switch (x; x) {
    case 1:
      ++x;
  }

  switch (int x, y = 0; y) {
    case 1:
      ++x;
    default:
      ++y;
  }

  switch (int x, y = 0; y) { // expected-note 2 {{previous definition is here}}
    case 0:
      int x = 0; // expected-error {{redefinition of 'x'}}
    case 1:
      int y = 0; // expected-error {{redefinition of 'y'}}
  };

  switch (x; int a = 0) {
    case 0:
      ++a;
  }

  switch (x, +x; int a = 0) { // expected-note {{previous definition is here}} expected-warning {{unused}}
    case 0:
      int a = 0; // expected-error {{redefinition of 'a'}} // expected-note {{previous definition is here}}
    case 1:
      int a = 0; // expected-error {{redefinition of 'a'}}
  }

  switch (int b = 0; b) {
    case 0:
      break;
  }
  b = 2; // expected-error {{use of undeclared identifier}}
}

constexpr bool constexpr_if_init(int n) {
  if (int a = n; ++a > 0)
    return true;
  else
    return false;
}

constexpr int constexpr_switch_init(int n) {
  switch (int p = n + 2; p) {
    case 0:
      return 0;
    case 1:
      return 1;
    default:
      return -1;
  }
}

void test_constexpr_init_stmt() {
  constexpr bool a = constexpr_if_init(-2);
  static_assert(!a, "");
  static_assert(constexpr_if_init(1), "");

  constexpr int b = constexpr_switch_init(-1);
  static_assert(b == 1, "");
  static_assert(constexpr_switch_init(-2) == 0, "");
  static_assert(constexpr_switch_init(-5) == -1, "");
}
