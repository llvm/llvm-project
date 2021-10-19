// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -fsyntax-only -fdouble-square-bracket-attributes -verify %s

[[clang::availability(macosx,introduced=10.4,deprecated=10.2)]] void f0(); // expected-warning{{feature cannot be deprecated in macOS version 10.2 before it was introduced in version 10.4; attribute ignored}}
[[clang::availability(ios,obsoleted=2.1,deprecated=3.0)]] void f1();  // expected-warning{{feature cannot be obsoleted in iOS version 2.1 before it was deprecated in version 3.0; attribute ignored}}
[[clang::availability(ios,introduced=2.1,deprecated=2.1)]] void f2();

[[clang::availability(macosx,introduced=8.0,deprecated=9.0, message="use CTFontCopyFullName")]]
extern void
ATSFontGetName(const char *oName); // expected-note {{'ATSFontGetName' has been explicitly marked deprecated here}}

void test_10095131() {
  ATSFontGetName("Hello"); // expected-warning {{'ATSFontGetName' is deprecated: first deprecated in macOS 9.0 - use CTFontCopyFullName}}
}

enum
[[clang::availability(macos, unavailable)]]
{
    NSDataWritingFileProtectionWriteOnly = 0x30000000,
    NSDataWritingFileProtectionCompleteUntilUserAuthentication = 0x40000000,
};

extern int x [[clang::availability(macosx,introduced=10.5)]];
extern int x;

int i [[clang::availability(this, should = 1.0)]]; // expected-error {{'should' is not an availability stage; use 'introduced', 'deprecated', or 'obsoleted'}} \
                                                   // expected-warning {{unknown platform 'this' in availability macro}}
