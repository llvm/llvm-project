// RUN: %clang_cc1 "-triple" "x86_64-apple-macos10.15" -darwin-target-variant-triple x86_64-apple-ios13.1-macabi -fsyntax-only -ftarget-variant-availability-checks -verify %s
// RUN: %clang_cc1 "-triple" x86_64-apple-ios13.1-macabi -darwin-target-variant-triple "x86_64-apple-macos10.15"  -fsyntax-only -ftarget-variant-availability-checks -verify -D INVERTED %s

__attribute__((availability(macos, introduced=10.10, deprecated=10.15), availability(ios, introduced=9, deprecated=12))) // expected-warning {{macOS availability is ignored without a valid 'SDKSettings.json' in the SDK}}
void bothDeprecated() { // expected-note {{'bothDeprecated' has been explicitly marked deprecated here}}
}

__attribute__((availability(macos, introduced=10.10, deprecated=10.15), availability(ios, introduced=9, deprecated=14)))
void macOSDeprecated() { // expected-note {{'macOSDeprecated' has been explicitly marked deprecated here}}
}

__attribute__((availability(macos, introduced=10.10, deprecated=11), availability(ios, introduced=9, deprecated=12)))
void iOSDeprecated() { // expected-note {{'iOSDeprecated' has been explicitly marked deprecated here}}
}

__attribute__((availability(macos, introduced=10.10, deprecated=11), availability(ios, introduced=13.0, deprecated=14.0)))
void bothNotDeprecated() {
}

void checkDeprecated() {
  // The compiler may warn here if the deployment target is older than the version introduced
  bothDeprecated();
#ifndef INVERTED
  // expected-warning@-2 {{'bothDeprecated' is deprecated: first deprecated in macOS 10.15 and first deprecated in macCatalyst 13.1}}
#else
  // expected-warning@-4 {{'bothDeprecated' is deprecated: first deprecated in macCatalyst 13.1 and first deprecated in macOS 10.15}}
#endif
  macOSDeprecated(); // expected-warning {{'macOSDeprecated' is deprecated: first deprecated in macOS 10.15}}
  iOSDeprecated();   // expected-warning {{'iOSDeprecated' is deprecated: first deprecated in macCatalyst 13.1}}
  bothNotDeprecated();
}

__attribute__((availability(macos, unavailable), availability(ios, unavailable)))
void bothUnavailable() { // expected-note {{'bothUnavailable' has been explicitly marked unavailable here}}
}

__attribute__((availability(macos, introduced=10.10), availability(ios, unavailable)))
void iosUnavailable() { // expected-note {{'iosUnavailable' has been explicitly marked unavailable here}}
}

__attribute__((availability(macos, unavailable), availability(ios, introduced=11)))
void macOSUnavailable() { // expected-note {{'macOSUnavailable' has been explicitly marked unavailable here}}
}

__attribute__((availability(macos, introduced=10.10), availability(ios, introduced=11)))
void bothAvailable() { }

__attribute__((availability(macos, introduced=10.10, obsoleted=10.15), availability(ios, introduced=9, obsoleted=12)))
void bothObsoleted() { // expected-note {{'bothObsoleted' has been explicitly marked unavailable here}}
}

void checkUnavailable() {
  bothUnavailable();
#ifndef INVERTED
  // expected-error@-2 {{'bothUnavailable' is unavailable: not available on macOS and not available on macCatalyst}}
#else
  // expected-error@-4 {{'bothUnavailable' is unavailable: not available on macCatalyst and not available on macOS}}
#endif
  iosUnavailable(); // expected-error {{'iosUnavailable' is unavailable: not available on macCatalyst}}
  macOSUnavailable(); // expected-error {{'macOSUnavailable' is unavailable: not available on macOS}}
  bothAvailable();
  
  bothObsoleted();
#ifndef INVERTED
  // expected-error@-2 {{'bothObsoleted' is unavailable: obsoleted in macOS 10.15 and obsoleted in macCatalyst 13.1}}
#else
  // expected-error@-4 {{'bothObsoleted' is unavailable: obsoleted in macCatalyst 13.1 and obsoleted in macOS 10.15}}
#endif
}

__attribute__((availability(macos, introduced=10.10, deprecated=10.15)))
typedef struct ZipperedTypedefDifferentOffendingDecl { // expected-note {{'ZipperedTypedefDifferentOffendingDecl' has been explicitly marked deprecated here}}
// expected-warning@-1 {{'ZipperedTypedefDifferentOffendingDecl' is deprecated: first deprecated in macCatalyst 13}}
// expected-note@-2 {{'ZipperedTypedefDifferentOffendingDecl' has been explicitly marked deprecated here}}
  int x;
} __attribute__((availability(ios, introduced=9, deprecated=12))) ZipperedTypedefDifferentOffendingDecl_t; // expected-note {{'ZipperedTypedefDifferentOffendingDecl_t' has been explicitly marked deprecated here}}

void checkZipperDiffOffendingDecl() {
  ZipperedTypedefDifferentOffendingDecl_t tt;
#ifndef INVERTED
  // expected-warning@-2 {{'ZipperedTypedefDifferentOffendingDecl_t' is deprecated: first deprecated in macOS 10.15 and first deprecated in macCatalyst 13.1}}
#else
  // expected-warning@-4 {{'ZipperedTypedefDifferentOffendingDecl_t' is deprecated: first deprecated in macCatalyst 13.1 and first deprecated in macOS 10.15}}
#endif
}

__attribute__((availability(macos, introduced=10.10, deprecated=10.15), availability(ios, unavailable)))
void mixUnavailableAndDeprecated() {
  // expected-note@-1 {{'mixUnavailableAndDeprecated' has been explicitly marked unavailable here}}
  // expected-note@-2 {{'mixUnavailableAndDeprecated' has been explicitly marked deprecated here}}
}

__attribute__((availability(macos, introduced=10.10, deprecated=11), availability(ios, unavailable)))
void iOSUnavailableMacGood() {
  // expected-note@-1 {{'iOSUnavailableMacGood' has been explicitly marked unavailable here}}
}

__attribute__((availability(macos, introduced=10.10, deprecated=10.15), availability(ios, introduced=11)))
void macOSDeprecatedIOSGood() {
  // expected-note@-1 {{'macOSDeprecatedIOSGood' has been explicitly marked deprecated here}}
}

__attribute__((availability(macos, introduced=10.10, deprecated=10.15, replacement="foo"), availability(ios, introduced=9, deprecated=12, replacement="bar")))
void bothDeprecatedDiffReplacement() { // expected-note {{'bothDeprecatedDiffReplacement' has been explicitly marked deprecated here}}
}

__attribute__((availability(macos, unavailable, replacement="bar"), availability(ios, unavailable, replacement="foo")))
void bothUnavailableSameReplacement() { // expected-note {{'bothUnavailableSameReplacement' has been explicitly marked unavailable here}}
}

__attribute__((availability(ios, unavailable)))
void justIosUnavailable() { // expected-note {{'justIosUnavailable' has been explicitly marked unavailable here}}
}

__attribute__((availability(macos, introduced=10.10, deprecated=10.15)))
void justMacOSDeprecated() { // expected-note {{'justMacOSDeprecated' has been explicitly marked deprecated here}}
}

void checkMix() {
  mixUnavailableAndDeprecated();
  // expected-error@-1 {{'mixUnavailableAndDeprecated' is unavailable: not available on macCatalyst}}
  // expected-warning@-2 {{'mixUnavailableAndDeprecated' is deprecated: first deprecated in macOS 10.15}}
  
  iOSUnavailableMacGood();
  // expected-error@-1 {{'iOSUnavailableMacGood' is unavailable: not available on macCatalyst}}
  
  macOSDeprecatedIOSGood();
  // expected-warning@-1 {{'macOSDeprecatedIOSGood' is deprecated: first deprecated in macOS 10.15}}

  bothDeprecatedDiffReplacement();
#ifndef INVERTED
  // expected-warning@-2 {{'bothDeprecatedDiffReplacement' is deprecated: first deprecated in macOS 10.15 and first deprecated in macCatalyst 13.1}}
#else
  // expected-warning@-4 {{'bothDeprecatedDiffReplacement' is deprecated: first deprecated in macCatalyst 13.1 and first deprecated in macOS 10.15}}
#endif

  bothUnavailableSameReplacement();
#ifndef INVERTED
  // expected-error@-2 {{'bothUnavailableSameReplacement' is unavailable: not available on macOS and not available on macCatalyst}}
#else
  // expected-error@-4 {{'bothUnavailableSameReplacement' is unavailable: not available on macCatalyst and not available on macOS}}
#endif

  justIosUnavailable(); // expected-error {{'justIosUnavailable' is unavailable: not available on macCatalyst}}

  justMacOSDeprecated(); // expected-warning {{'justMacOSDeprecated' is deprecated: first deprecated in macOS 10.15}}
}

int deprecatedFunc() __attribute__((deprecated)); // expected-note {{'deprecatedFunc' has been explicitly marked deprecated here}}
int unavailFunc() __attribute__((unavailable)); // expected-note {{'unavailFunc' has been explicitly marked unavailable here}}

void a() {
  int (*ptr)() = deprecatedFunc; // expected-warning {{'deprecatedFunc' is deprecated}}
  int (*ptr2)() = unavailFunc; // expected-error {{'unavailFunc' is unavailable}}
}


struct IntroducedLaterBoth { } __attribute__((availability(macos, introduced=11), availability(ios, introduced=14)));
#ifndef INVERTED
// expected-note@-2 {{'IntroducedLaterBoth' has been marked as being introduced in macOS 11 here, but the deployment target is macOS 10.15}}
#else
// expected-note@-4 {{'IntroducedLaterBoth' has been marked as being introduced in macCatalyst 14 here, but the deployment target is macCatalyst 13.1}}
#endif

struct NotYetIntroduced { // expected-note {{annotate 'NotYetIntroduced' with an availability attribute to silence this warning}}
  struct IntroducedLaterBoth x;
#ifndef INVERTED
  // expected-warning@-2 {{'IntroducedLaterBoth' is only available on macOS 11 and macCatalyst 14 or newer}}
#else
  // expected-warning@-4 {{'IntroducedLaterBoth' is only available on macCatalyst 14 and macOS 11 or newer}}
#endif
};
