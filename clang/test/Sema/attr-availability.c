// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -fsyntax-only -verify %s

void f0() __attribute__((availability(macosx,introduced=10.4,deprecated=10.2))); // expected-warning{{feature cannot be deprecated in OS X version 10.2 before it was introduced in version 10.4; attribute ignored}}
void f1() __attribute__((availability(ios,obsoleted=2.1,deprecated=3.0)));  // expected-warning{{feature cannot be obsoleted in iOS version 2.1 before it was deprecated in version 3.0; attribute ignored}}
void f2() __attribute__((availability(ios,introduced=2.1,deprecated=2.1)));

void f3() __attribute__((availability(otheros,introduced=2.2))); // expected-warning{{unknown platform 'otheros' in availability macro}}

// rdar://10095131
extern void 
ATSFontGetName(const char *oName) __attribute__((availability(macosx,introduced=8.0,deprecated=9.0, message="use CTFontCopyFullName"))); // expected-note {{'ATSFontGetName' declared here}}

extern void
ATSFontGetPostScriptName(int flags) __attribute__((availability(macosx,introduced=8.0,obsoleted=9.0, message="use ATSFontGetFullPostScriptName"))); // expected-note {{function has been explicitly marked unavailable here}}

void test_10095131() {
  ATSFontGetName("Hello"); // expected-warning {{'ATSFontGetName' is deprecated: first deprecated in OS X 9.0 - use CTFontCopyFullName}}
  ATSFontGetPostScriptName(100); // expected-error {{'ATSFontGetPostScriptName' is unavailable: obsoleted in OS X 9.0 - use ATSFontGetFullPostScriptName}}
}

// rdar://10711037
__attribute__((availability(macos, unavailable))) // expected-warning {{attribute 'availability' is ignored}}
enum {
    NSDataWritingFileProtectionWriteOnly = 0x30000000,
    NSDataWritingFileProtectionCompleteUntilUserAuthentication = 0x40000000,
};

void f4(int) __attribute__((availability(ios,deprecated=3.0)));
void f4(int) __attribute__((availability(ios,introduced=4.0))); // expected-warning {{feature cannot be deprecated in iOS version 3.0 before it was introduced in version 4.0; attribute ignored}}

void f5(int) __attribute__((availability(ios,deprecated=3.0),
                            availability(ios,introduced=4.0)));  // expected-warning {{feature cannot be deprecated in iOS version 3.0 before it was introduced in version 4.0; attribute ignored}}

void f6(int) __attribute__((availability(ios,deprecated=3.0))); // expected-note {{previous attribute is here}}
void f6(int) __attribute__((availability(ios,deprecated=4.0))); // expected-warning {{availability does not match previous declaration}}

void f7(int) __attribute__((availability(ios,introduced=2.0)));
void f7(int) __attribute__((availability(ios,deprecated=3.0))); // expected-note {{previous attribute is here}}
void f7(int) __attribute__((availability(ios,deprecated=4.0))); // expected-warning {{availability does not match previous declaration}}


// <rdar://problem/11886458>
#if !__has_feature(attribute_availability_with_message)
# error "Missing __has_feature"
#endif
