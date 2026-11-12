// RUN: %clang_cc1 "-triple" "x86_64-apple-ios13.1-macabi" -fsyntax-only -verify %s

void f0(void) __attribute__((availability(macOS, introduced = 10.11)));
// expected-warning@-1 {{macOS availability is ignored without a valid 'SDKSettings.json' in the SDK}}
void f1(void) __attribute__((availability(macOS, introduced = 10.15)));
