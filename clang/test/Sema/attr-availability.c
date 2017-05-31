// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -fsyntax-only -fblocks -verify %s
// RUN: %clang_cc1 -D WARN_PARTIAL -Wpartial-availability -triple x86_64-apple-darwin9 -fsyntax-only -fblocks -verify %s
// 

void f0() __attribute__((availability(macosx,introduced=10.4,deprecated=10.2))); // expected-warning{{feature cannot be deprecated in macOS version 10.2 before it was introduced in version 10.4; attribute ignored}}
void f1() __attribute__((availability(ios,obsoleted=2.1,deprecated=3.0)));  // expected-warning{{feature cannot be obsoleted in iOS version 2.1 before it was deprecated in version 3.0; attribute ignored}}
void f2() __attribute__((availability(ios,introduced=2.1,deprecated=2.1)));

void f3() __attribute__((availability(otheros,introduced=2.2))); // expected-warning{{unknown platform 'otheros' in availability macro}}

// rdar://10095131
extern void 
ATSFontGetName(const char *oName) __attribute__((availability(macosx,introduced=8.0,deprecated=9.0, message="use CTFontCopyFullName"))); // expected-note {{'ATSFontGetName' has been explicitly marked deprecated here}}

extern void
ATSFontGetPostScriptName(int flags) __attribute__((availability(macosx,introduced=8.0,obsoleted=9.0, message="use ATSFontGetFullPostScriptName"))); // expected-note {{'ATSFontGetPostScriptName' has been explicitly marked unavailable here}}

#if defined(WARN_PARTIAL)
// expected-note@+3 {{has been explicitly marked partial here}}
#endif
extern void
PartiallyAvailable() __attribute__((availability(macosx,introduced=10.8)));

enum __attribute__((availability(macosx,introduced=10.8))) PartialEnum {
  kPartialEnumConstant,
};

void test_10095131() {
  ATSFontGetName("Hello"); // expected-warning {{'ATSFontGetName' is deprecated: first deprecated in macOS 9.0 - use CTFontCopyFullName}}
  ATSFontGetPostScriptName(100); // expected-error {{'ATSFontGetPostScriptName' is unavailable: obsoleted in macOS 9.0 - use ATSFontGetFullPostScriptName}}

#if defined(WARN_PARTIAL)
  // expected-warning@+2 {{is only available on macOS 10.8 or newer}} expected-note@+2 {{enclose 'PartiallyAvailable' in an @available check to silence this warning}}
#endif
  PartiallyAvailable();
}

extern void PartiallyAvailable() ;
void with_redeclaration() {
  PartiallyAvailable();  // Don't warn.

  // enums should never warn.
  enum PartialEnum p = kPartialEnumConstant;
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

extern int x __attribute__((availability(macosx,introduced=10.5)));
extern int x;

void f8() {
  int (^b)(int);
  b = ^ (int i) __attribute__((availability(macosx,introduced=10.2))) { return 1; }; // expected-warning {{'availability' attribute ignored}}
}

extern int x2 __attribute__((availability(macosx,introduced=10.2))); // expected-note {{previous attribute is here}}
extern int x2 __attribute__((availability(macosx,introduced=10.5))); // expected-warning {{availability does not match previous declaration}}


enum Original {
  OriginalDeprecated __attribute__((availability(macosx, deprecated=10.2))), // expected-note + {{'OriginalDeprecated' has been explicitly marked deprecated here}}
  OriginalUnavailable __attribute__((availability(macosx, unavailable))) // expected-note + {{'OriginalUnavailable' has been explicitly marked unavailable here}}
};

enum AllDeprecated {
  AllDeprecatedCase, // expected-note + {{'AllDeprecatedCase' has been explicitly marked deprecated here}}
  AllDeprecatedUnavailable __attribute__((availability(macosx, unavailable))) // expected-note + {{'AllDeprecatedUnavailable' has been explicitly marked unavailable here}}
} __attribute__((availability(macosx, deprecated=10.2)));

enum AllUnavailable {
  AllUnavailableCase, // expected-note + {{'AllUnavailableCase' has been explicitly marked unavailable here}}
} __attribute__((availability(macosx, unavailable)));

enum User {
  UserOD = OriginalDeprecated, // expected-warning {{deprecated}}
  UserODDeprecated __attribute__((availability(macosx, deprecated=10.2))) = OriginalDeprecated,
  UserODUnavailable __attribute__((availability(macosx, unavailable))) = OriginalDeprecated,

  UserOU = OriginalUnavailable, // expected-error {{unavailable}}
  UserOUDeprecated __attribute__((availability(macosx, deprecated=10.2))) = OriginalUnavailable, // expected-error {{unavailable}}
  UserOUUnavailable __attribute__((availability(macosx, unavailable))) = OriginalUnavailable,

  UserAD = AllDeprecatedCase, // expected-warning {{deprecated}}
  UserADDeprecated __attribute__((availability(macosx, deprecated=10.2))) = AllDeprecatedCase,
  UserADUnavailable __attribute__((availability(macosx, unavailable))) = AllDeprecatedCase,

  UserADU = AllDeprecatedUnavailable, // expected-error {{unavailable}}
  UserADUDeprecated __attribute__((availability(macosx, deprecated=10.2))) = AllDeprecatedUnavailable, // expected-error {{unavailable}}
  UserADUUnavailable __attribute__((availability(macosx, unavailable))) = AllDeprecatedUnavailable,

  UserAU = AllUnavailableCase, // expected-error {{unavailable}}
  UserAUDeprecated __attribute__((availability(macosx, deprecated=10.2))) = AllUnavailableCase, // expected-error {{unavailable}}
  UserAUUnavailable __attribute__((availability(macosx, unavailable))) = AllUnavailableCase,
};

enum UserDeprecated {
  UserDeprecatedOD = OriginalDeprecated,
  UserDeprecatedODDeprecated __attribute__((availability(macosx, deprecated=10.2))) = OriginalDeprecated,
  UserDeprecatedODUnavailable __attribute__((availability(macosx, unavailable))) = OriginalDeprecated,

  UserDeprecatedOU = OriginalUnavailable, // expected-error {{unavailable}}
  UserDeprecatedOUDeprecated __attribute__((availability(macosx, deprecated=10.2))) = OriginalUnavailable, // expected-error {{unavailable}}
  UserDeprecatedOUUnavailable __attribute__((availability(macosx, unavailable))) = OriginalUnavailable,

  UserDeprecatedAD = AllDeprecatedCase,
  UserDeprecatedADDeprecated __attribute__((availability(macosx, deprecated=10.2))) = AllDeprecatedCase,
  UserDeprecatedADUnavailable __attribute__((availability(macosx, unavailable))) = AllDeprecatedCase,

  UserDeprecatedADU = AllDeprecatedUnavailable, // expected-error {{unavailable}}
  UserDeprecatedADUDeprecated __attribute__((availability(macosx, deprecated=10.2))) = AllDeprecatedUnavailable, // expected-error {{unavailable}}
  UserDeprecatedADUUnavailable __attribute__((availability(macosx, unavailable))) = AllDeprecatedUnavailable,

  UserDeprecatedAU = AllUnavailableCase, // expected-error {{unavailable}}
  UserDeprecatedAUDeprecated __attribute__((availability(macosx, deprecated=10.2))) = AllUnavailableCase, // expected-error {{unavailable}}
  UserDeprecatedAUUnavailable __attribute__((availability(macosx, unavailable))) = AllUnavailableCase,
} __attribute__((availability(macosx, deprecated=10.2)));

enum UserUnavailable {
  UserUnavailableOD = OriginalDeprecated,
  UserUnavailableODDeprecated __attribute__((availability(macosx, deprecated=10.2))) = OriginalDeprecated,
  UserUnavailableODUnavailable __attribute__((availability(macosx, unavailable))) = OriginalDeprecated,

  UserUnavailableOU = OriginalUnavailable,
  UserUnavailableOUDeprecated __attribute__((availability(macosx, deprecated=10.2))) = OriginalUnavailable,
  UserUnavailableOUUnavailable __attribute__((availability(macosx, unavailable))) = OriginalUnavailable,

  UserUnavailableAD = AllDeprecatedCase,
  UserUnavailableADDeprecated __attribute__((availability(macosx, deprecated=10.2))) = AllDeprecatedCase,
  UserUnavailableADUnavailable __attribute__((availability(macosx, unavailable))) = AllDeprecatedCase,

  UserUnavailableADU = AllDeprecatedUnavailable,
  UserUnavailableADUDeprecated __attribute__((availability(macosx, deprecated=10.2))) = AllDeprecatedUnavailable,
  UserUnavailableADUUnavailable __attribute__((availability(macosx, unavailable))) = AllDeprecatedUnavailable,

  UserUnavailableAU = AllUnavailableCase,
  UserUnavailableAUDeprecated __attribute__((availability(macosx, deprecated=10.2))) = AllUnavailableCase,
  UserUnavailableAUUnavailable __attribute__((availability(macosx, unavailable))) = AllUnavailableCase,
} __attribute__((availability(macosx, unavailable)));
