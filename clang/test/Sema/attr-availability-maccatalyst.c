// RUN: %clang_cc1 "-triple" "x86_64-apple-ios13.1-macabi" -fsyntax-only -verify %s
// RUN: %clang_cc1 "-triple" "x86_64-apple-ios13.1-macabi" -fapplication-extension -D APPEXT -fsyntax-only -verify %s

#ifdef APPEXT

#define maccatalyst maccatalyst_app_extension
#define macCatalyst maccatalyst_app_extension
#define ios ios_app_extension

#endif

void f0(int) __attribute__((availability(maccatalyst,introduced=2.0,deprecated=9.1))); // expected-note {{'f0' has been explicitly marked deprecated here}}
void f1(int) __attribute__((availability(maccatalyst,introduced=2.1)));
void f2(int) __attribute__((availability(macCatalyst,introduced=2.0,deprecated=9.0))); // expected-note {{'f2' has been explicitly marked deprecated here}}
void f3(int) __attribute__((availability(maccatalyst,introduced=3.0, obsoleted=9.0))); // expected-note {{'f3' has been explicitly marked unavailable here}}
void f32(int) __attribute__((availability(maccatalyst,introduced=3.0, obsoleted=9.0))); // expected-note {{'f32' has been explicitly marked unavailable here}}


void f5(int) __attribute__((availability(maccatalyst,introduced=2.0))) __attribute__((availability(maccatalyst,deprecated=9.0))); // expected-note {{'f5' has been explicitly marked deprecated here}}
void f6(int) __attribute__((availability(maccatalyst,deprecated=9.0))); // expected-note {{'f6' has been explicitly marked deprecated here}}
void f6(int) __attribute__((availability(macCatalyst,introduced=2.0)));

void f7(void) // expected-note {{'f7' has been explicitly marked deprecated here}}
__attribute__((availability(maccatalyst,introduced=3.0, deprecated=4.0)))
__attribute__((availability(ios,introduced=2.0, deprecated=5.0)));

void f8(void) // expected-note {{'f8' has been explicitly marked unavailable here}}
__attribute__((availability(maccatalyst,introduced=3.0, obsoleted=4.0)))
__attribute__((availability(ios,introduced=2.0, obsoleted=5.0)));

void f9(void) // expected-note {{'f9' has been explicitly marked unavailable here}}
__attribute__((availability(maccatalyst,unavailable)))
__attribute__((availability(ios,introduced=2.0)));

void test(void) {
  f0(0);
#ifndef APPEXT
  // expected-warning@-2 {{'f0' is deprecated: first deprecated in macCatalyst 9.1}}
#else
  // expected-warning@-4 {{'f0' is deprecated: first deprecated in macCatalyst (App Extension) 9.1}}
#endif
  f1(0);
  f2(0);
#ifndef APPEXT
  // expected-warning@-2 {{'f2' is deprecated: first deprecated in macCatalyst 9.0}}
#else
  // expected-warning@-4 {{'f2' is deprecated: first deprecated in macCatalyst (App Extension) 9.0}}
#endif
  f3(0);
#ifndef APPEXT
  // expected-error@-2 {{'f3' is unavailable: obsoleted in macCatalyst 9.0}}
#else
  // expected-error@-4 {{'f3' is unavailable: obsoleted in macCatalyst (App Extension) 9.0}}
#endif
  f32(0);
#ifndef APPEXT
  // expected-error@-2 {{'f32' is unavailable: obsoleted in macCatalyst 9.0}}
#else
  // expected-error@-4 {{'f32' is unavailable: obsoleted in macCatalyst (App Extension) 9.0}}
#endif
  f5(0); // expected-warning{{'f5' is deprecated: first deprecated in macCatalyst}}
  f6(0); // expected-warning{{'f6' is deprecated: first deprecated in macCatalyst}}

  f7();
#ifndef APPEXT
  // expected-warning@-2 {{'f7' is deprecated: first deprecated in macCatalyst 4.0}}
#else
  // expected-warning@-4 {{'f7' is deprecated: first deprecated in macCatalyst (App Extension) 4.0}}
#endif
  f8();
#ifndef APPEXT
  // expected-error@-2 {{'f8' is unavailable: obsoleted in macCatalyst 4.0}}
#else
  // expected-error@-4 {{'f8' is unavailable: obsoleted in macCatalyst (App Extension) 4.0}}
#endif
  f9(); // expected-error {{'f9' is unavailable}}
}

// Don't inherit "deprecated"/"obsoleted" from iOS for Mac Catalyst.

void f100(void)
__attribute__((availability(maccatalyst,introduced=3.0)))
__attribute__((availability(ios,introduced=2.0, deprecated=5.0)));

void f101(void)
__attribute__((availability(maccatalyst,introduced=3.0)))
__attribute__((availability(ios,introduced=2.0, obsoleted=5.0)));

void f102(void)
__attribute__((availability(maccatalyst,introduced=3.0)))
__attribute__((availability(ios,unavailable)));

void f103(void)
__attribute__((availability(ios,unavailable)));

void f103(void)
__attribute__((availability(maccatalyst,introduced=3.0)));

void dontInheritObsoletedDeprecated(void) {
  f100();
  f101();
  f102();
  f103();
}

// Inherit the ios availability when Mac Catalyst isn't given.

void f202(void) __attribute__((availability(ios,introduced=2.0, deprecated=5.0))); // expected-note {{here}}
void f203(void) __attribute__((availability(ios,introduced=2.0, obsoleted=5.0))); // expected-note {{here}}
void f204(void) __attribute__((availability(ios,unavailable))); // expected-note {{here}}

void inheritIosAvailability(void) {
  f202();
#ifndef APPEXT
// expected-warning@-2 {{'f202' is deprecated: first deprecated in macCatalyst 13.1}}
#else
// expected-warning@-4 {{'f202' is deprecated: first deprecated in macCatalyst (App Extension) 13.1}}
#endif
  f203();
#ifndef APPEXT
  // expected-error@-2 {{'f203' is unavailable: obsoleted in macCatalyst 13.1}}
#else
  // expected-error@-4 {{'f203' is unavailable: obsoleted in macCatalyst (App Extension) 13.1}}
#endif
  f204();
#ifndef APPEXT
  // expected-error@-2 {{'f204' is unavailable: not available on macCatalyst}}
#else
  // expected-error@-4 {{'f204' is unavailable: not available on macCatalyst (App Extension)}}
#endif
}
