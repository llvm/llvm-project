// Test availability attributes are enforced for Fuchsia targets.

// REQUIRES: x86-registered-target
// RUN: %clang_cc1 "-triple" "x86_64-unknown-fuchsia" -ffuchsia-api-level=16 -fsyntax-only -verify %s
// RUN: %clang_cc1 "-triple" "x86_64-unknown-fuchsia" -fsyntax-only %s 2>&1 | FileCheck %s

// If the version is not specified, we should not get any errors since there
// is no checking (the major version number is zero).
// CHECK-NOT: error:

void f0(int) __attribute__((availability(fuchsia, introduced = 14, deprecated = 19)));
void f1(int) __attribute__((availability(fuchsia, introduced = 16)));
void f2(int) __attribute__((availability(fuchsia, introduced = 14, deprecated = 16))); // expected-note {{'f2' has been explicitly marked deprecated here}}
void f3(int) __attribute__((availability(fuchsia, introduced = 19, strict))); // expected-note {{'f3' has been explicitly marked unavailable here}}
void f4(int) __attribute__((availability(fuchsia, introduced = 9, deprecated = 11, obsoleted = 16), availability(ios, introduced = 2.0, deprecated = 3.0))); // expected-note{{explicitly marked unavailable}}
void f5(int) __attribute__((availability(ios, introduced = 3.2), availability(fuchsia, unavailable)));                                                       // expected-note{{'f5' has been explicitly marked unavailable here}}
void f6(int) __attribute__((availability(fuchsia, introduced = 16.0)));                                                                                      // expected-warning {{Fuchsia API Level prohibits specifying a minor or sub-minor version}}
void f7(int) __attribute__((availability(fuchsia, introduced = 16.1))); // expected-warning {{Fuchsia API Level prohibits specifying a minor or sub-minor version}}
void f8(int) __attribute__((availability(fuchsia, introduced = 19))); // nothing will happen as 'strict' is not specified.

void test() {
  f0(0);
  f1(0);
  f2(0); // expected-warning{{'f2' is deprecated: first deprecated in Fuchsia 16}}
  f3(0); // expected-error{{'f3' is unavailable: introduced in Fuchsia 19}}
  f4(0); // expected-error{{f4' is unavailable: obsoleted in Fuchsia 16}}
  f5(0); // expected-error{{'f5' is unavailable: not available on Fuchsia}}
  f8(0);
}
