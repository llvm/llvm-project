// RUN: %clang_cc1 -fobjc-arc -verify -Wno-objc-root-class %s

// rdar://problem/9150784
void test(void) {
  __weak id x; // expected-error {{cannot create __weak reference because the current deployment target does not support weak references}}
  __weak void *v; // expected-warning {{'__weak' only applies to Objective-C object or block pointer types}}
}

@interface A
@property (weak) id testObjectWeakProperty; // expected-note {{declared here}}
@end

@implementation A
// rdar://9605088
@synthesize testObjectWeakProperty; // expected-error {{cannot synthesize weak property because the current deployment target does not support weak references}}
@end
