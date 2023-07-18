// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify -fblocks -Wno-objc-root-class %s
// RUN: %clang_cc1 -x objective-c++ -triple x86_64-apple-darwin10 -fsyntax-only -verify -fblocks -Wno-objc-root-class %s

// rdar://10681443
@interface I
- (void) compileSandboxProfileAndReturnError:(__attribute__((__blocks__(byref))) id)errorp; // expected-error {{__block attribute not allowed, only allowed on local variables}}
@end

@implementation I
- (void) compileSandboxProfileAndReturnError:(__attribute__((__blocks__(byref))) id)errorp {} // expected-error {{__block attribute not allowed, only allowed on local variables}}
@end

