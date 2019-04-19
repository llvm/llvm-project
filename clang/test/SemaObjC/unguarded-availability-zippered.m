// RUN: %clang_cc1 -triple x86_64-apple-macos10.14 -darwin-target-variant-triple x86_64-apple-ios12-macabi -fblocks -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-apple-ios12-macabi -darwin-target-variant-triple x86_64-apple-macos10.14 -fblocks -fsyntax-only -verify -D INVERTED %s

// XFAIL: *

#define AVAILABLE_PREV_MAC __attribute__((availability(macos, introduced = 10.13)))
#define AVAILABLE_CURRENT_MAC __attribute__((availability(macos, introduced = 10.14)))
#define AVAILABLE_NEXT_MAC __attribute__((availability(macos, introduced = 10.15)))

#define AVAILABLE_PREV_IOS __attribute__((availability(ios, introduced = 11)))
#define AVAILABLE_CURRENT_IOS __attribute__((availability(iOSMac, introduced = 12)))
#define AVAILABLE_NEXT_IOS __attribute__((availability(ios, introduced = 13)))

void bothPreviouslyAvailable() AVAILABLE_PREV_MAC AVAILABLE_PREV_IOS;
void bothCurrentlyAvailable() AVAILABLE_CURRENT_MAC AVAILABLE_CURRENT_IOS;
void bothWillBeAvailable() AVAILABLE_NEXT_MAC AVAILABLE_NEXT_IOS;
#ifndef INVERTED
// expected-note@-2 3 {{'bothWillBeAvailable' has been marked as being introduced in macOS 10.15 here, but the deployment target is macOS 10.14.0}}
// expected-note@-3 {{'bothWillBeAvailable' has been marked as being introduced in UIKit for macOS 14 here, but the deployment target is UIKit for macOS 13.0.0}}
#else
// expected-note@-5 3 {{'bothWillBeAvailable' has been marked as being introduced in UIKit for macOS 14 here, but the deployment target is UIKit for macOS 13.0.0}}
// expected-note@-6 {{'bothWillBeAvailable' has been marked as being introduced in macOS 10.15 here, but the deployment target is macOS 10.14.0}}
#endif

void macOSCurrentlyAvailable() AVAILABLE_CURRENT_MAC AVAILABLE_NEXT_IOS;
// #ifndef INVERTED
// expected-note@-2 2 {{'macOSCurrentlyAvailable' has been marked as being introduced in UIKit for macOS 14 here, but the deployment target is UIKit for macOS 13.0.0}}
// #else
// #endif

void macOSNextAvailableiOSNotAvailable() AVAILABLE_NEXT_MAC __attribute__((availability(ios, unavailable)));
// expected-note@-1 2 {{'macOSNextAvailableiOSNotAvailable' has been explicitly marked unavailable here}}
// expected-note@-2 {{'macOSNextAvailableiOSNotAvailable' has been marked as being introduced in macOS 10.15 here, but the deployment target is macOS 10.14.0}}

void test() {
  bothPreviouslyAvailable();
  bothCurrentlyAvailable();
  bothWillBeAvailable(); // expected-note{{enclose 'bothWillBeAvailable' in an @available check to silence this warning}}
#ifndef INVERTED
  // expected-warning@-2 {{'bothWillBeAvailable' is only available on macOS 10.15 and UIKit for macOS 13 or newer}}
#else
  // expected-warning@-4 {{'bothWillBeAvailable' is only available on UIKit for macOS 13 and macOS 10.15 or newer}}
#endif
  
  
  macOSCurrentlyAvailable(); // expected-warning {{'macOSCurrentlyAvailable' is only available on UIKit for macOS 13 or newer}}
  // expected-note@-1 {{enclose 'macOSCurrentlyAvailable' in an @available check to silence this warning}}

  if (@available(ios 13, macos 10.15, *))
    bothWillBeAvailable();
  if (@available(ios 13, *))
    bothWillBeAvailable(); // expected-warning {{'bothWillBeAvailable' is only available on macOS 10.15 or newer}}
  // expected-note@-1 {{enclose}}
  if (@available(ios 12, *))
    bothWillBeAvailable(); // expected-note {{enclose}}
#ifndef INVERTED
  // expected-warning@-2 {{'bothWillBeAvailable' is only available on macOS 10.15 and UIKit for macOS 13 or newer}}
#else
  // expected-warning@-4 {{'bothWillBeAvailable' is only available on UIKit for macOS 13 and macOS 10.15 or newer}}
#endif
  if (@available(macos 10.15, *))
    bothWillBeAvailable(); // expected-warning {{'bothWillBeAvailable' is only available on UIKit for macOS 13 or newer}}
  // expected-note@-1 {{enclose}}

  if (@available(macos 10.15, *))
    macOSCurrentlyAvailable(); // expected-warning {{'macOSCurrentlyAvailable' is only available on UIKit for macOS 13 or newer}}
  // expected-note@-1 {{enclose}}
  if (@available(ios 13, *))
    macOSCurrentlyAvailable();

  macOSNextAvailableiOSNotAvailable();
  // expected-error@-1 {{'macOSNextAvailableiOSNotAvailable' is unavailable: not available on UIKit for macOS}}
  // expected-warning@-2 {{'macOSNextAvailableiOSNotAvailable' is only available on macOS 10.15 or newer}}
  // expected-note@-3 {{enclose}}

  if (@available(macos 10.15, *))
    macOSNextAvailableiOSNotAvailable(); // expected-error {{'macOSNextAvailableiOSNotAvailable' is unavailable: not available on UIKit for macOS}}
}
