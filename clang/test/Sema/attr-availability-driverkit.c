// RUN: %clang_cc1 "-triple" "x86_64-apple-driverkit20.0" -fsyntax-only -verify %s

void f0(int) __attribute__((availability(driverkit,introduced=19.0,deprecated=20.0))); // expected-note {{'f0' has been explicitly marked deprecated here}}
void f1(int) __attribute__((availability(driverkit,introduced=20.0)));
void f2(int) __attribute__((availability(driverkit,introduced=19.0,deprecated=20.0))); // expected-note {{'f2' has been explicitly marked deprecated here}}
void f3(int) __attribute__((availability(driverkit,introduced=20.0)));
void f4(int) __attribute__((availability(macosx,introduced=10.1,deprecated=10.3,obsoleted=10.5), availability(driverkit,introduced=19.0,deprecated=19.5,obsoleted=20.0))); // expected-note{{explicitly marked unavailable}}

void f5(int) __attribute__((availability(driverkit,introduced=19.0))) __attribute__((availability(driverkit,deprecated=20.0))); // expected-note {{'f5' has been explicitly marked deprecated here}}
void f6(int) __attribute__((availability(driverkit,deprecated=20.0))); // expected-note {{'f6' has been explicitly marked deprecated here}}
void f7(int) __attribute__((availability(driverkit,introduced=19.0)));

void test() {
  f0(0); // expected-warning{{'f0' is deprecated: first deprecated in DriverKit 20.0}}
  f1(0);
  f2(0); // expected-warning{{'f2' is deprecated: first deprecated in DriverKit 20.0}}
  f3(0);
  f4(0); // expected-error{{f4' is unavailable: obsoleted in DriverKit 20.0}}
  f5(0); // expected-warning{{'f5' is deprecated: first deprecated in DriverKit 20.0}}
  f6(0); // expected-warning{{'f6' is deprecated: first deprecated in DriverKit 20.0}}
  f7(0);
}
