// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -fsyntax-only -verify -Wno-objc-root-class %s
// expected-no-diagnostics
// rdar://9651605
// rdar://12958191

@interface Foo
@property (getter=getVal) int val __attribute__((unavailable));
@property (getter=getVal) int val2 __attribute__((availability(macosx,unavailable)));
- Method __attribute__((unavailable));
+ CMethod __attribute__((unavailable));
@end

@implementation Foo
@end

