// RUN: %clang_cc1 -triple x86_64-apple-macos10.15 -darwin-target-variant-triple x86_64-apple-ios13.1-macabi -fblocks -fsyntax-only -Wno-ignored-availability-without-sdk-settings -verify %s

#define AVAILABLE_CUR_MAC __attribute__((availability(macos, introduced = 10.15)))
#define AVAILABLE_NEXT_IOS __attribute__((availability(ios, introduced = 14)))

void macCatalystIntroducedLater() AVAILABLE_CUR_MAC AVAILABLE_NEXT_IOS;

void test() {
  macCatalystIntroducedLater();
}

// expected-no-diagnostics
