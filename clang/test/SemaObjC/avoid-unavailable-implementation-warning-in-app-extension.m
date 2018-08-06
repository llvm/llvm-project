// RUN: %clang_cc1 -triple arm64-apple-ios11 -fapplication-extension -Wdeprecated-implementations -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -triple arm64-apple-tvos11 -fapplication-extension -Wdeprecated-implementations -verify -Wno-objc-root-class %s
// Declarations marked as 'unavailable' in an app extension should not generate a
// warning on implementation.

@interface Parent
- (void)ok __attribute__((availability(ios_app_extension,unavailable,message="not available")));
- (void)reallyUnavail __attribute__((availability(ios,unavailable))); // expected-note {{method 'reallyUnavail' declared here}}
- (void)reallyUnavail2 __attribute__((unavailable)); // expected-note {{method 'reallyUnavail2' declared here}}
@end

@interface Child : Parent
@end

@implementation Child

- (void)ok { // no warning.
}
- (void)reallyUnavail { // expected-warning {{implementing unavailable method}}
}
- (void)reallyUnavail2 { // expected-warning {{implementing unavailable method}}
}

@end
