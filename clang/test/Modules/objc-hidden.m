// RUN: rm -rf %t
// RUN: %clang_cc1 -fsyntax-only -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -I %S/Inputs/objc-hidden/System -F %S/Inputs/objc-hidden -verify -x objective-c %s
// expected-no-diagnostics

// Make sure we don't crash with hidden decls.
@import FakeUnavailableObjCFramework;

@implementation UnavailableObjCClass
- (void)someMethod { }
@end

