// RUN: %clang_cc1 "-triple" "x86_64-apple-ios3.0" -fsyntax-only -verify %s

void f0(int) __attribute__((availability(ios,introduced=2.0,deprecated=2.1))); // expected-note {{'f0' has been explicitly marked deprecated here}}
void f1(int) __attribute__((availability(ios,introduced=2.1)));
void f2(int) __attribute__((availability(iOS,introduced=2.0,deprecated=3.0))); // expected-note {{'f2' has been explicitly marked deprecated here}}
void f3(int) __attribute__((availability(ios,introduced=3.0)));
void f4(int) __attribute__((availability(macosx,introduced=10.1,deprecated=10.3,obsoleted=10.5), availability(ios,introduced=2.0,deprecated=2.1,obsoleted=3.0))); // expected-note{{explicitly marked unavailable}}

void f5(int) __attribute__((availability(ios,introduced=2.0))) __attribute__((availability(ios,deprecated=3.0))); // expected-note {{'f5' has been explicitly marked deprecated here}}
void f6(int) __attribute__((availability(ios,deprecated=3.0))); // expected-note {{'f6' has been explicitly marked deprecated here}}
void f6(int) __attribute__((availability(iOS,introduced=2.0)));

void test() {
  f0(0); // expected-warning{{'f0' is deprecated: first deprecated in iOS 2.1}}
  f1(0);
  f2(0); // expected-warning{{'f2' is deprecated: first deprecated in iOS 3.0}}
  f3(0);
  f4(0); // expected-error{{f4' is unavailable: obsoleted in iOS 3.0}}
  f5(0); // expected-warning{{'f5' is deprecated: first deprecated in iOS 3.0}}
  f6(0); // expected-warning{{'f6' is deprecated: first deprecated in iOS 3.0}}
}
