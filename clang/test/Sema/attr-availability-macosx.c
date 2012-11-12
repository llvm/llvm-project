// RUN: %clang_cc1 "-triple" "x86_64-apple-darwin9.0.0" -fsyntax-only -verify %s

void f0(int) __attribute__((availability(macosx,introduced=10.4,deprecated=10.6)));
void f1(int) __attribute__((availability(macosx,introduced=10.5)));
void f2(int) __attribute__((availability(macosx,introduced=10.4,deprecated=10.5))); // expected-note {{'f2' declared here}}
void f3(int) __attribute__((availability(macosx,introduced=10.6)));
void f4(int) __attribute__((availability(macosx,introduced=10.1,deprecated=10.3,obsoleted=10.5), availability(ios,introduced=2.0,deprecated=3.0))); // expected-note{{explicitly marked unavailable}}
void f5(int) __attribute__((availability(ios,introduced=3.2), availability(macosx,unavailable))); // expected-note{{function has been explicitly marked unavailable here}}

void test() {
  f0(0);
  f1(0);
  f2(0); // expected-warning{{'f2' is deprecated: first deprecated in OS X 10.5}}
  f3(0);
  f4(0); // expected-error{{f4' is unavailable: obsoleted in OS X 10.5}}
  f5(0); // expected-error{{'f5' is unavailable: not available on OS X}}
}

// rdar://10535640

enum {
    foo __attribute__((availability(macosx,introduced=8.0,deprecated=9.0)))
};

enum {
    bar __attribute__((availability(macosx,introduced=8.0,deprecated=9.0))) = foo
};

enum __attribute__((availability(macosx,introduced=8.0,deprecated=9.0))) {
    bar1 = foo
};
