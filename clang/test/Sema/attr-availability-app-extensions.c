// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9.0 -fsyntax-only -fapplication-extension %s -verify
// RUN: %clang_cc1 -triple armv7-apple-ios9.0 -fsyntax-only -fapplication-extension %s -verify
// RUN: %clang_cc1 -triple arm64-apple-tvos3.0 -fsyntax-only -fapplication-extension -DTVOS=1 -verify %s
// RUN: %clang_cc1 -triple arm64-apple-tvos3.0 -fsyntax-only -fapplication-extension -verify %s

#if __has_feature(attribute_availability_app_extension)
 __attribute__((availability(macosx_app_extension,unavailable)))
#ifndef TVOS
 __attribute__((availability(ios_app_extension,unavailable)))
#else
 __attribute__((availability(tvos_app_extension,unavailable)))
#endif
#endif
void f0(int); // expected-note {{'f0' has been explicitly marked unavailable here}}

__attribute__((availability(macosx,unavailable)))
#ifndef TVOS
__attribute__((availability(ios,unavailable)))
#else
  __attribute__((availability(tvos,unavailable)))
#endif
void f1(int); // expected-note {{'f1' has been explicitly marked unavailable here}}

void test() {
  f0(1); // expected-error {{'f0' is unavailable: not available on}}
  f1(1); // expected-error {{'f1' is unavailable}}
}

