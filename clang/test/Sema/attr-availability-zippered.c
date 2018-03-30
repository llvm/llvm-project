// RUN: %clang_cc1 "-triple" "x86_64-apple-macos10.14" -darwin-target-variant-triple x86_64-apple-ios12-macabi -fsyntax-only -verify %s

// XFAIL: *
__attribute__((availability(macos, introduced=10.10, deprecated=10.14), availability(ios, introduced=9, deprecated=12)))
void bothDeprecated() { // expected-note {{'bothDeprecated' has been explicitly marked deprecated here}}
}

__attribute__((availability(macos, introduced=10.10, deprecated=10.14), availability(ios, introduced=9, deprecated=13)))
void macOSDeprecated() { // expected-note {{'macOSDeprecated' has been explicitly marked deprecated here}}
}

__attribute__((availability(macos, introduced=10.10, deprecated=10.15), availability(ios, introduced=9, deprecated=12)))
void iOSDeprecated() { // expected-note {{'iOSDeprecated' has been explicitly marked deprecated here}}
}

__attribute__((availability(macos, introduced=10.10, deprecated=10.15), availability(ios, introduced=9, deprecated=13)))
void bothNotDeprecated() {
}

void checkDeprecated() {
  // The compiler may warn here if the deployment target is older than the version introduced
  bothDeprecated();  // expected-warning {{'bothDeprecated' is deprecated: first deprecated in macOS 10.14 and first deprecated in iOS (on macOS) 12}}
  macOSDeprecated(); // expected-warning {{'macOSDeprecated' is deprecated: first deprecated in macOS 10.14}}
  iOSDeprecated();   // expected-warning {{'iOSDeprecated' is deprecated: first deprecated in iOS (on macOS) 12}}
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

__attribute__((availability(macos, introduced=10.10, obsoleted=10.14), availability(ios, introduced=9, obsoleted=12)))
void bothObsoleted() { // expected-note {{'bothObsoleted' has been explicitly marked unavailable here}}
}

void checkUnavailable() {
  bothUnavailable();  // expected-error {{'bothUnavailable' is unavailable: not available on macOS and not available on iOS (on macOS)}}
  iosUnavailable(); // expected-error {{'iosUnavailable' is unavailable: not available on iOS (on macOS)}}
  macOSUnavailable(); // expected-error {{'macOSUnavailable' is unavailable: not available on macOS}}
  bothAvailable();
  
  bothObsoleted(); // expected-error {{'bothObsoleted' is unavailable: obsoleted in macOS 10.14 and obsoleted in iOS (on macOS) 12}}
}

__attribute__((availability(macos, introduced=10.10, deprecated=10.14)))
typedef struct ZipperedTypedefDifferentOffendingDecl { // expected-note {{'ZipperedTypedefDifferentOffendingDecl' has been explicitly marked deprecated here}}
// expected-warning@-1 {{'ZipperedTypedefDifferentOffendingDecl' is deprecated: first deprecated in iOS (on macOS) 12}}
// expected-note@-2 {{'ZipperedTypedefDifferentOffendingDecl' has been explicitly marked deprecated here}}
  int x;
} __attribute__((availability(ios, introduced=9, deprecated=12))) ZipperedTypedefDifferentOffendingDecl_t; // expected-note {{'ZipperedTypedefDifferentOffendingDecl_t' has been explicitly marked deprecated here}}

void checkZipperDiffOffendingDecl() {
  ZipperedTypedefDifferentOffendingDecl_t tt; // expected-warning {{'ZipperedTypedefDifferentOffendingDecl_t' is deprecated: first deprecated in macOS 10.14 and first deprecated in iOS (on macOS) 12}}
}

__attribute__((availability(macos, introduced=10.10, deprecated=10.14), availability(ios, unavailable)))
void mixUnavailableAndDeprecated() {
  // expected-note@-1 {{'mixUnavailableAndDeprecated' has been explicitly marked unavailable here}}
  // expected-note@-2 {{'mixUnavailableAndDeprecated' has been explicitly marked deprecated here}}
}

__attribute__((availability(macos, introduced=10.10, deprecated=10.15), availability(ios, unavailable)))
void iOSUnavailableMacGood() {
  // expected-note@-1 {{'iOSUnavailableMacGood' has been explicitly marked unavailable here}}
}

__attribute__((availability(macos, introduced=10.10, deprecated=10.14), availability(ios, introduced=11)))
void macOSDeprecatedIOSGood() {
  // expected-note@-1 {{'macOSDeprecatedIOSGood' has been explicitly marked deprecated here}}
}

__attribute__((availability(macos, introduced=10.10, deprecated=10.14, replacement="foo"), availability(ios, introduced=9, deprecated=12, replacement="bar")))
void bothDeprecatedDiffReplacement() { // expected-note {{'bothDeprecatedDiffReplacement' has been explicitly marked deprecated here}}
}

__attribute__((availability(macos, unavailable, replacement="bar"), availability(ios, unavailable, replacement="foo")))
void bothUnavailableSameReplacement() { // expected-note {{'bothUnavailableSameReplacement' has been explicitly marked unavailable here}}
}

__attribute__((availability(ios, unavailable)))
void justIosUnavailable() { // expected-note {{'justIosUnavailable' has been explicitly marked unavailable here}}
}

__attribute__((availability(macos, introduced=10.10, deprecated=10.14)))
void justMacOSDeprecated() { // expected-note {{'justMacOSDeprecated' has been explicitly marked deprecated here}}
}

void checkMix() {
  mixUnavailableAndDeprecated();
  // expected-error@-1 {{'mixUnavailableAndDeprecated' is unavailable: not available on iOS (on macOS)}}
  // expected-warning@-2 {{'mixUnavailableAndDeprecated' is deprecated: first deprecated in macOS 10.14}}
  
  iOSUnavailableMacGood();
  // expected-error@-1 {{'iOSUnavailableMacGood' is unavailable: not available on iOS (on macOS)}}
  
  macOSDeprecatedIOSGood();
  // expected-warning@-1 {{'macOSDeprecatedIOSGood' is deprecated: first deprecated in macOS 10.14}}

  bothDeprecatedDiffReplacement(); // expected-warning {{'bothDeprecatedDiffReplacement' is deprecated: first deprecated in macOS 10.14 and first deprecated in iOS (on macOS) 12}}
  
  bothUnavailableSameReplacement(); // expected-error {{'bothUnavailableSameReplacement' is unavailable: not available on macOS and not available on iOS (on macOS)}}
  
  justIosUnavailable(); // expected-error {{'justIosUnavailable' is unavailable: not available on iOS (on macOS)}}
  
  justMacOSDeprecated(); // expected-warning {{'justMacOSDeprecated' is deprecated: first deprecated in macOS 10.14}}
}

int deprecatedFunc() __attribute__((deprecated)); // expected-note {{'deprecatedFunc' has been explicitly marked deprecated here}}
int unavailFunc() __attribute__((unavailable)); // expected-note {{'unavailFunc' has been explicitly marked unavailable here}}

void a() {
  int (*ptr)() = deprecatedFunc; // expected-warning {{'deprecatedFunc' is deprecated}}
  int (*ptr2)() = unavailFunc; // expected-error {{'unavailFunc' is unavailable}}
}


struct IntroducedLaterBoth { } __attribute__((availability(macos, introduced=10.15), availability(ios, introduced=13)));
// expected-note@-1 {{'IntroducedLaterBoth' has been explicitly marked partial here}}

struct NotYetIntroduced { // expected-note {{annotate 'NotYetIntroduced' with an availability attribute to silence this warning}}
  struct IntroducedLaterBoth x; // expected-warning {{'IntroducedLaterBoth' is only available on macOS 10.15 and iOS (on macOS) 13 or newer}}
};
