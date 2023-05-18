// RUN: %clang_cc1 -fsyntax-only -std=c++1z -verify %s

void f(int n) {
  switch (n) {
  case 0:
    n += 1;
    [[fallthrough]]; // ok
  case 1:
    if (n) {
      [[fallthrough]]; // ok
    } else {
      return;
    }
  case 2:
    for (int n = 0; n != 10; ++n)
      [[fallthrough]]; // expected-error {{does not directly precede switch label}}
  case 3:
    while (true)
      [[fallthrough]]; // expected-error {{does not directly precede switch label}}
  case 4:
    while (false)
      [[fallthrough]]; // expected-error {{does not directly precede switch label}}
  case 5:
    do [[fallthrough]]; while (true); // expected-error {{does not directly precede switch label}}
  case 6:
    do [[fallthrough]]; while (false); // expected-error {{does not directly precede switch label}}
  case 7:
    switch (n) {
    case 0:
      // FIXME: This should be an error, even though the next thing we do is to
      // fall through in an outer switch statement.
      [[fallthrough]];
    }
  case 8:
    [[fallthrough]]; // expected-error {{does not directly precede switch label}}
    goto label;
  label:
  case 9:
    n += 1;
  case 10: // no warning, -Wimplicit-fallthrough is not enabled in this test, and does not need to
           // be enabled for these diagnostics to be produced.
    break;
  }
}

[[fallthrough]] typedef int n; // expected-error {{'fallthrough' attribute cannot be applied to a declaration}}
typedef int [[fallthrough]] n; // expected-error {{'fallthrough' attribute cannot be applied to types}}
typedef int n [[fallthrough]]; // expected-error {{'fallthrough' attribute cannot be applied to a declaration}}

enum [[fallthrough]] E {}; // expected-error {{'fallthrough' attribute cannot be applied to a declaration}}
class [[fallthrough]] C {}; // expected-error {{'fallthrough' attribute cannot be applied to a declaration}}

[[fallthrough]] // expected-error {{'fallthrough' attribute cannot be applied to a declaration}}
void g() {
  [[fallthrough]] int n; // expected-error {{'fallthrough' attribute cannot be applied to a declaration}}
  [[fallthrough]] ++n; // expected-error {{'fallthrough' attribute only applies to empty statements}}

  switch (n) {
    // FIXME: This should be an error.
    [[fallthrough]];
    return;

  case 0:
    [[fallthrough, fallthrough]]; // ok
  case 1:
    [[fallthrough(0)]]; // expected-error {{argument list}}
  case 2:
    break;
  }
}
