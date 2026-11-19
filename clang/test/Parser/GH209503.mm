// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

// Method and function bodies are parsed when their @implementation ends; make
// sure they are queued into the right @implementation (if any) after a nested
// container has ended one.

@interface Z
@end
@interface A
@end

@implementation Z
namespace N {
@implementation A // expected-error {{Objective-C declarations may only appear in global scope}}
@end
}
- (void)m {
  undeclared(); // expected-error {{use of undeclared identifier 'undeclared'}}
}
@end

@implementation NSArray // expected-warning {{cannot find interface declaration for 'NSArray'}} \
                        // expected-note {{implementation started here}}
@interface NSIndexSet // expected-error {{missing '@end'}} \
                      // expected-note {{class started here}}
// expected-warning@+1 {{function definition inside an Objective-C container is deprecated}}
void f(void) {} // expected-error {{missing '@end'}}
