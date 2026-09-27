// RUN: %clang_cc1 -triple arm64-apple-xros1 -verify=ios -isysroot %S/Inputs/XROS1.0.sdk %s 2>&1
// RUN: %clang_cc1 -triple arm64-apple-xros1 -fapplication-extension -verify=ios,ext -isysroot %S/Inputs/XROS1.0.sdk %s 2>&1

// RUN: %clang_cc1 -triple arm64-apple-xros2 -DXROS2 -verify=ios -isysroot %S/Inputs/XROS1.0.sdk  %s 2>&1

__attribute__((availability(ios, unavailable)))
void ios_unavail(); // ios-note {{}}

__attribute__((availability(ios_app_extension, unavailable)))
void ios_ext_unavail(); // ext-note {{}}

void use() {
  ios_unavail(); // ios-error {{'ios_unavail' is unavailable: not available on }}
  ios_ext_unavail(); // ext-error {{'ios_ext_unavail' is unavailable: not available on }}
}

__attribute__((availability(ios, introduced=10)))
void ios_introduced_10();

__attribute__((availability(ios_app_extension, introduced=10)))
void ios_ext_introduced_10();

__attribute__((availability(ios, introduced=17.1)))
void ios_introduced_17();

__attribute__((availability(ios_app_extension, introduced=17.1)))
void ios_ext_introduced_17();

__attribute__((availability(ios, introduced=18)))
void ios_introduced_18(); // ios-note {{}}

__attribute__((availability(ios_app_extension, introduced=18)))
void ios_ext_introduced_18(); // ext-note {{}}

void useIntroduced() {
  // introduced iOS < 10 => introduced xrOS 1
  ios_introduced_10();
  ios_ext_introduced_10();
  // introduced iOS 17.1 => introduced xrOS 1
  ios_introduced_17();
  ios_ext_introduced_17();
  // introduced iOS 18 => xros unavailable (no mapping)
  ios_introduced_18(); // ios-error {{is unavailable: not available on }}
  ios_ext_introduced_18(); // ext-error {{is unavailable: not available on }}
}

__attribute__((availability(ios, deprecated=10)))
void ios_deprecated_10(); // ios-note {{}}

__attribute__((availability(ios_app_extension, deprecated=10)))
void ios_ext_deprecated_10(); // ext-note {{}}

__attribute__((availability(ios, deprecated=17.1)))
void ios_deprecated_17(); // ios-note {{}}

__attribute__((availability(ios_app_extension, deprecated=17.1)))
void ios_ext_deprecated_17(); // ext-note {{}}

__attribute__((availability(ios, deprecated=18)))
void ios_deprecated_18();
#ifdef XROS2
// ios-note@-2 {{}}
#endif

__attribute__((availability(ios_app_extension, deprecated=18)))
void ios_ext_deprecated_18();

void useDeprecated() {
  // deprecated iOS < 10 => deprecated xrOS 1
  ios_deprecated_10(); // ios-warning {{is deprecated: first deprecated in}}
  ios_ext_deprecated_10(); // ext-warning {{is deprecated: first deprecated in}}
  // deprecated iOS 17.1 => deprecated xrOS 1
  ios_deprecated_17(); // ios-warning {{is deprecated: first deprecated in}}
  ios_ext_deprecated_17(); // ext-warning {{is deprecated: first deprecated in}}
  // deprecated iOS 18 => deprecated xrOS 1.0.99
  ios_deprecated_18();
#ifdef XROS2
  // ios-warning@-2 {{is deprecated: first deprecated in}}
#endif
  ios_ext_deprecated_18();
}

__attribute__((availability(ios, obsoleted=10)))
void ios_obsoleted_10(); // ios-note {{}}

__attribute__((availability(ios_app_extension, obsoleted=10)))
void ios_ext_obsoleted_10(); // ext-note {{}}

__attribute__((availability(ios, obsoleted=17.1)))
void ios_obsoleted_17(); // ios-note {{}}

__attribute__((availability(ios_app_extension, obsoleted=17.1)))
void ios_ext_obsoleted_17(); // ext-note {{}}

__attribute__((availability(ios, obsoleted=18)))
void ios_obsoleted_18();
#ifdef XROS2
// ios-note@-2 {{}}
#endif

__attribute__((availability(ios_app_extension, obsoleted=18)))
void ios_ext_obsoleted_18();

void useObsoleted() {
  // deprecated iOS < 10 => deprecated xrOS 1
  ios_obsoleted_10(); // ios-error {{is unavailable: obsoleted in}}
  ios_ext_obsoleted_10(); // ext-error {{is unavailable: obsoleted in}}
  // deprecated iOS 17.1 => deprecated xrOS 1
  ios_obsoleted_17(); // ios-error {{is unavailable: obsoleted in}}
  ios_ext_obsoleted_17(); // ext-error {{is unavailable: obsoleted in}}
  // obsoleted iOS 18 => obsoleted xrOS 1.0.99
  ios_obsoleted_18();
#ifdef XROS2
  // ios-error@-2 {{is unavailable: obsoleted in}}
#endif
  ios_ext_obsoleted_18();
}
