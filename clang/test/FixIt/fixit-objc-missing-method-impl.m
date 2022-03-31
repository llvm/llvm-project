// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: cp %s %t
// RUN: not %clang_cc1 -pedantic -Werror -fixit -x objective-c %t
// RUN: %clang_cc1 -pedantic -Werror -x objective-c %t

__attribute__((objc_root_class))
@interface NSObject
@end

@interface Foo : NSObject
- (void)fooey;  // expected-note{{method 'fooey' declared here}}
@end

@implementation Foo  // expected-warning{{method definition for 'fooey' not found}}
@end
