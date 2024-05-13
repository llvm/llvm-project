// RUN: %clang_cc1 "-triple" "x86_64-apple-ios13.1-macabi" -isysroot %S/Inputs/MacOSX11.0.sdk -fsyntax-only -verify %s
// RUN: %clang_cc1 "-triple" "x86_64-apple-ios14-macabi" -isysroot %S/Inputs/MacOSX11.0.sdk -DIOS14 -fsyntax-only -verify %s

void f0(void) __attribute__((availability(macOS, introduced = 10.11)));
void f1(void) __attribute__((availability(macOS, introduced = 10.15)));
void f2(void) __attribute__(( // expected-note {{'f2' has been explicitly marked deprecated here}}
    availability(macOS, introduced = 10.11,
                 deprecated = 10.12)));
void f3(void)
    __attribute__((availability(macOS, introduced = 10.11, deprecated = 10.14)))
    __attribute__((availability(iOS, introduced = 11.0)));

void f4(void)
__attribute__((availability(macOS, introduced = 10, deprecated = 100000)));

void fAvail(void) __attribute__((availability(macOS, unavailable)));

void f16(void) __attribute__((availability(macOS, introduced = 11.0)));
#ifndef IOS14
// expected-note@-2 {{here}}
#endif

void fObs(void) __attribute__((availability(macOS, introduced = 10.11, obsoleted = 10.15))); // expected-note {{'fObs' has been explicitly marked unavailable here}}

void fAPItoDepr(void) __attribute__((availability(macOS, introduced = 10.11, deprecated = 100000)));

void dontRemapFutureVers(void) __attribute__((availability(macOS, introduced = 20)));

void usage(void) {
  f0();
  f1();
  f2(); // expected-warning {{'f2' is deprecated: first deprecated in macCatalyst 13.1}}
  f3();
  f4();
  fAvail();
  f16();
#ifndef IOS14
  // expected-warning@-2 {{'f16' is only available on macCatalyst 14.0 or newer}} expected-note@-2 {{enclose}}
#endif
  fObs(); // expected-error {{'fObs' is unavailable: obsoleted in macCatalyst 13.1}}
  fAPItoDepr();
  dontRemapFutureVers();
}

#ifdef IOS14

void f15_4(void) __attribute__((availability(macOS, introduced = 10.15, deprecated = 10.15.4))); // expected-note {{here}}
void f15_3(void) __attribute__((availability(macOS, introduced = 10.15, deprecated = 10.15.3))); // expected-note {{here}}
void f15_2(void) __attribute__((availability(macOS, introduced = 10.15, deprecated = 10.15.2))); // expected-note {{here}}

void usage16(void) {
  f15_2(); // expected-warning {{'f15_2' is deprecated: first deprecated in macCatalyst 13.3}}
  f15_3(); // expected-warning {{'f15_3' is deprecated: first deprecated in macCatalyst 13.3.1}}
  f15_4(); // expected-warning {{'f15_4' is deprecated: first deprecated in macCatalyst 13.4}}
  f16();
}

#endif
