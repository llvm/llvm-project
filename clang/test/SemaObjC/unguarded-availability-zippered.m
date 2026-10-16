// RUN: %clang_cc1 -triple x86_64-apple-macos10.15 -darwin-target-variant-triple x86_64-apple-ios13.1-macabi -fblocks -fsyntax-only -ftarget-variant-availability-checks -Wno-ignored-availability-without-sdk-settings -verify %s
// RUN: %clang_cc1 -triple x86_64-apple-ios13.1-macabi -darwin-target-variant-triple x86_64-apple-macos10.15 -fblocks -fsyntax-only -ftarget-variant-availability-checks -Wno-ignored-availability-without-sdk-settings -verify -D INVERTED %s


#define AVAILABLE_PREV_MAC __attribute__((availability(macos, introduced = 10.13)))
#define AVAILABLE_CURRENT_MAC __attribute__((availability(macos, introduced = 10.15)))
#define AVAILABLE_NEXT_MAC __attribute__((availability(macos, introduced = 11.0)))

#define AVAILABLE_PREV_IOS __attribute__((availability(macCatalyst, introduced = 11)))
#define AVAILABLE_CURRENT_IOS __attribute__((availability(macCatalyst, introduced = 13.1)))
#define AVAILABLE_NEXT_IOS __attribute__((availability(macCatalyst, introduced = 14)))

void bothPreviouslyAvailable() AVAILABLE_PREV_MAC AVAILABLE_PREV_IOS;
void bothCurrentlyAvailable() AVAILABLE_CURRENT_MAC AVAILABLE_CURRENT_IOS;
void bothWillBeAvailable() AVAILABLE_NEXT_MAC AVAILABLE_NEXT_IOS;
#ifndef INVERTED
// expected-note@-2 3 {{'bothWillBeAvailable' has been marked as being introduced in macOS 11.0 here, but the deployment target is macOS 10.15}}
// expected-note@-3 {{'bothWillBeAvailable' has been marked as being introduced in macCatalyst 14 here, but the deployment target is macCatalyst 13.1}}
#else
// expected-note@-5 3 {{'bothWillBeAvailable' has been marked as being introduced in macCatalyst 14 here, but the deployment target is macCatalyst 13.1}}
// expected-note@-6 {{'bothWillBeAvailable' has been marked as being introduced in macOS 11.0 here, but the deployment target is macOS 10.15}}
#endif

void macOSCurrentlyAvailable() AVAILABLE_CURRENT_MAC AVAILABLE_NEXT_IOS;
// #ifndef INVERTED
// expected-note@-2 2 {{'macOSCurrentlyAvailable' has been marked as being introduced in macCatalyst 14 here, but the deployment target is macCatalyst 13.1}}
// #else
// #endif

void macOSNextAvailableiOSNotAvailable() AVAILABLE_NEXT_MAC __attribute__((availability(ios, unavailable)));
// expected-note@-1 2 {{'macOSNextAvailableiOSNotAvailable' has been explicitly marked unavailable here}}
// expected-note@-2 {{'macOSNextAvailableiOSNotAvailable' has been marked as being introduced in macOS 11.0 here, but the deployment target is macOS 10.15}}

void test() {
  bothPreviouslyAvailable();
  bothCurrentlyAvailable();
  bothWillBeAvailable(); // expected-note{{enclose 'bothWillBeAvailable' in an @available check to silence this warning}}
#ifndef INVERTED
  // expected-warning@-2 {{'bothWillBeAvailable' is only available on macOS 11.0 and macCatalyst 14 or newer}}
#else
  // expected-warning@-4 {{'bothWillBeAvailable' is only available on macCatalyst 14 and macOS 11.0 or newer}}
#endif
  
  
  macOSCurrentlyAvailable(); // expected-warning {{'macOSCurrentlyAvailable' is only available on macCatalyst 14 or newer}}
  // expected-note@-1 {{enclose 'macOSCurrentlyAvailable' in an @available check to silence this warning}}

  if (@available(ios 14, macos 11.0, *))
    bothWillBeAvailable();
  if (@available(ios 14, *))
    bothWillBeAvailable(); // expected-warning {{'bothWillBeAvailable' is only available on macOS 11.0 or newer}}
  // expected-note@-1 {{enclose}}
  if (@available(ios 13, *))
    bothWillBeAvailable(); // expected-note {{enclose}}
#ifndef INVERTED
  // expected-warning@-2 {{'bothWillBeAvailable' is only available on macOS 11.0 and macCatalyst 14 or newer}}
#else
  // expected-warning@-4 {{'bothWillBeAvailable' is only available on macCatalyst 14 and macOS 11.0 or newer}}
#endif
  if (@available(macos 11.0, *))
    bothWillBeAvailable(); // expected-warning {{'bothWillBeAvailable' is only available on macCatalyst 14 or newer}}
  // expected-note@-1 {{enclose}}

  if (@available(macos 11.0, *))
    macOSCurrentlyAvailable(); // expected-warning {{'macOSCurrentlyAvailable' is only available on macCatalyst 14 or newer}}
  // expected-note@-1 {{enclose}}
  if (@available(ios 14, *))
    macOSCurrentlyAvailable();

  macOSNextAvailableiOSNotAvailable();
  // expected-error@-1 {{'macOSNextAvailableiOSNotAvailable' is unavailable: not available on macCatalyst}}
  // expected-warning@-2 {{'macOSNextAvailableiOSNotAvailable' is only available on macOS 11.0 or newer}}
  // expected-note@-3 {{enclose}}

  if (@available(macos 11.0, *))
    macOSNextAvailableiOSNotAvailable(); // expected-error {{'macOSNextAvailableiOSNotAvailable' is unavailable: not available on macCatalyst}}
}
