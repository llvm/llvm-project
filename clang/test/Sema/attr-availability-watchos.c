// RUN: %clang_cc1 "-triple" "arm64-apple-watchos3.0" -fsyntax-only -verify %s

void f0(int) __attribute__((availability(ios,introduced=2.0,deprecated=2.1))); // expected-note {{'f0' has been explicitly marked deprecated here}}
void f1(int) __attribute__((availability(ios,introduced=2.1)));
void f2(int) __attribute__((availability(ios,introduced=2.0,deprecated=3.0))); // expected-note {{'f2' has been explicitly marked deprecated here}}
void f3(int) __attribute__((availability(ios,introduced=3.0)));
void f4(int) __attribute__((availability(macosx,introduced=10.1,deprecated=10.3,obsoleted=10.5), availability(ios,introduced=2.0,deprecated=2.1,obsoleted=3.0))); // expected-note{{explicitly marked unavailable}}
void f5(int) __attribute__((availability(ios,introduced=2.0))) __attribute__((availability(ios,deprecated=3.0))); // expected-note {{'f5' has been explicitly marked deprecated here}}
void f6(int) __attribute__((availability(ios,deprecated=12.1))); // OK - not deprecated for watchOS
void f7(int) __attribute__((availability(ios,deprecated=8.3))); // expected-note {{'f7' has been explicitly marked deprecated here}}
void f8(int) __attribute__((availability(ios,introduced=2.0,obsoleted=10.0))); // expected-note {{explicitly marked unavailable}}

void test() {
  f0(0); // expected-warning{{'f0' is deprecated: first deprecated in watchOS 2.0}}
  f1(0);
  f2(0); // expected-warning{{'f2' is deprecated: first deprecated in watchOS 2.0}}
  f3(0);
  f4(0); // expected-error {{f4' is unavailable: obsoleted in watchOS 2.0}}
  f5(0); // expected-warning {{'f5' is deprecated: first deprecated in watchOS 2.0}}
  f6(0);
  f7(0); // expected-warning {{'f7' is deprecated: first deprecated in watchOS 2.0}}
  f8(0); // expected-error {{'f8' is unavailable: obsoleted in watchOS 3.0}}
}

// Test watchOS specific attributes.
void f0_watchos(int) __attribute__((availability(watchos,introduced=2.0,deprecated=2.1))); // expected-note {{'f0_watchos' has been explicitly marked deprecated here}}
void f1_watchos(int) __attribute__((availability(watchos,introduced=2.1)));
void f2_watchos(int) __attribute__((availability(watchOS,introduced=2.0,deprecated=3.0))); // expected-note {{'f2_watchos' has been explicitly marked deprecated here}}
void f3_watchos(int) __attribute__((availability(watchos,introduced=3.0)));
void f4_watchos(int) __attribute__((availability(macosx,introduced=10.1,deprecated=10.3,obsoleted=10.5), availability(watchos,introduced=2.0,deprecated=2.1,obsoleted=3.0))); // expected-note{{explicitly marked unavailable}}
void f5_watchos(int) __attribute__((availability(watchos,introduced=2.0))) __attribute__((availability(ios,deprecated=3.0)));
void f5_attr_reversed_watchos(int) __attribute__((availability(ios, deprecated=3.0))) __attribute__((availability(watchos,introduced=2.0)));
void f5b_watchos(int) __attribute__((availability(watchos,introduced=2.0))) __attribute__((availability(watchos,deprecated=3.0))); // expected-note {{'f5b_watchos' has been explicitly marked deprecated here}}
void f5c_watchos(int) __attribute__((availability(ios,introduced=2.0))) __attribute__((availability(ios,deprecated=3.0))); // expected-note {{'f5c_watchos' has been explicitly marked deprecated here}}
void f6_watchos(int) __attribute__((availability(watchos,deprecated=3.0)));
void f6_watchos(int) __attribute__((availability(watchOS,introduced=2.0))); // expected-note {{'f6_watchos' has been explicitly marked deprecated here}}

void test_watchos() {
  f0_watchos(0); // expected-warning{{'f0_watchos' is deprecated: first deprecated in watchOS 2.1}}
  f1_watchos(0);
  f2_watchos(0); // expected-warning{{'f2_watchos' is deprecated: first deprecated in watchOS 3.0}}
  f3_watchos(0);
  f4_watchos(0); // expected-error{{'f4_watchos' is unavailable: obsoleted in watchOS 3.0}}
  // We get no warning here because any explicit 'watchos' availability causes
  // the ios availability to not implicitly become 'watchos' availability.  Otherwise we'd get
  // a deprecated warning.
  f5_watchos(0); // no-warning
  f5_attr_reversed_watchos(0); // no-warning
  // We get a deprecated warning here because both attributes are explicitly 'watchos'.
  f5b_watchos(0); // expected-warning {{'f5b_watchos' is deprecated: first deprecated in watchOS 3.0}}
  // We get a deprecated warning here because both attributes are 'ios' (both get mapped to 'watchos').
  f5c_watchos(0); // expected-warning {{'f5c_watchos' is deprecated: first deprecated in watchOS 2.0}}
  f6_watchos(0); // expected-warning {{'f6_watchos' is deprecated: first deprecated in watchOS 3.0}}
}
