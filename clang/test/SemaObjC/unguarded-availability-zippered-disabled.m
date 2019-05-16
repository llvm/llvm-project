// RUN: %clang_cc1 -triple x86_64-apple-macos10.14 -darwin-target-variant-triple x86_64-apple-ios13.1-macabi -fblocks -fsyntax-only -verify %s

#define AVAILABLE_CUR_MAC __attribute__((availability(macos, introduced = 10.14)))
#define AVAILABLE_NEXT_IOS __attribute__((availability(ios, introduced = 14)))

void uikitformacIntroducedLater() AVAILABLE_CUR_MAC AVAILABLE_NEXT_IOS;
// expected-warning@-1 {{macOS availability is ignored without a valid 'SDKSettings.json' in the SDK}}

void test() {
  uikitformacIntroducedLater();
}

