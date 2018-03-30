// RUN: %clang_cc1 -triple x86_64-apple-macos10.14 -darwin-target-variant-triple x86_64-apple-ios12-macabi -fblocks -fsyntax-only -verify %s

// XFAIL: *
#ifdef NO_WARNING
  // expected-no-diagnostics
#endif


#define AVAILABLE_PREV_MAC __attribute__((availability(macos, introduced = 10.13)))
#define AVAILABLE_CURRENT_MAC __attribute__((availability(macos, introduced = 10.14)))
#define AVAILABLE_NEXT_MAC __attribute__((availability(macos, introduced = 10.15)))

#define AVAILABLE_PREV_IOS __attribute__((availability(ios, introduced = 11)))
#define AVAILABLE_CURRENT_IOS __attribute__((availability(iOSMac, introduced = 12)))
#define AVAILABLE_NEXT_IOS __attribute__((availability(ios, introduced = 13)))

void bothPreviouslyAvailable() AVAILABLE_PREV_MAC AVAILABLE_PREV_IOS;
void bothCurrentlyAvailable() AVAILABLE_CURRENT_MAC AVAILABLE_CURRENT_IOS;
void bothWillBeAvailable() AVAILABLE_NEXT_MAC AVAILABLE_NEXT_IOS;
// expected-note@-1 3 {{'bothWillBeAvailable' has been explicitly marked partial here}}

void macOSCurrentlyAvailable() AVAILABLE_CURRENT_MAC AVAILABLE_NEXT_IOS;
// expected-note@-1 {{'macOSCurrentlyAvailable' has been explicitly marked partial here}}

void test() {
  bothPreviouslyAvailable();
  bothCurrentlyAvailable();
  bothWillBeAvailable(); // expected-warning {{'bothWillBeAvailable' is only available on macOS 10.15 and iOS (on macOS) 13 or newer}}
  // expected-note@-1{{enclose 'bothWillBeAvailable' in an @available check to silence this warning}}
  
  macOSCurrentlyAvailable(); // expected-warning {{'macOSCurrentlyAvailable' is only available on iOS (on macOS) 13 or newer}}
  // expected-note@-1 {{enclose 'macOSCurrentlyAvailable' in an @available check to silence this warning}}
  
  if (@available(ios 13, macos 10.15, *))
    bothWillBeAvailable();
  if (@available(ios 13, *))
    bothWillBeAvailable(); // expected-warning {{'bothWillBeAvailable' is only available on macOS 10.15 or newer}}
  // expected-note@-1 {{enclose}}
  if (@available(macos 10.15, *))
    bothWillBeAvailable(); // expected-warning {{'bothWillBeAvailable' is only available on iOS (on macOS) 13 or newer}}
  // expected-note@-1 {{enclose}}
}
