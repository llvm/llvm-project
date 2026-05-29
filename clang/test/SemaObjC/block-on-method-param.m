// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify -fblocks -Wno-objc-root-class %s
// RUN: %clang_cc1 -x objective-c++ -triple x86_64-apple-darwin10 -fsyntax-only -verify -fblocks -Wno-objc-root-class %s

@interface I
- (void) compileSandboxProfileAndReturnError:(__attribute__((__blocks__(byref))) id)errorp; // expected-error {{'__block' is not allowed on a nonlocal variable}}
@end

@implementation I
- (void) compileSandboxProfileAndReturnError:(__attribute__((__blocks__(byref))) id)errorp {} // expected-error {{'__block' is not allowed on a nonlocal variable}}
@end

