// RUN: %clang_cc1 -triple x86_64-apple-macosx10.14.0 -fsyntax-only -Wno-objc-root-class -verify %s

@interface I1 @end

// expected-warning@+1 {{'always_inline' attribute only applies to functions}}
__attribute__((always_inline))
@implementation I1 @end

// expected-warning@+1 {{'always_inline' attribute only applies to functions}}
__attribute__((always_inline))
@implementation I1 (MyCat) @end

// expected-warning@+1 {{'always_inline' attribute only applies to functions}}
__attribute__((always_inline))
// expected-warning@+1 {{cannot find interface declaration for 'I2'}}
@implementation I2 @end

// expected-error@+1 {{only applies to Objective-C interfaces}}
__attribute__((objc_root_class))
// expected-warning@+1 {{cannot find interface declaration for 'I3'}}
@implementation I3 @end

#define AVAIL_ATTR __attribute__((availability(macos, introduced=1000)))

typedef int AVAIL_ATTR unavail_int; // expected-note {{marked as being introduced}}

@interface I4 @end // expected-note {{annotate}}
@implementation I4 {
  unavail_int x; // expected-warning {{'unavail_int' is only available on macOS 1000 or newer}}
}
@end

@interface I5 @end

#pragma clang attribute push (AVAIL_ATTR, apply_to=objc_implementation)
@implementation I5 {
  unavail_int x;
}
@end
#pragma clang attribute pop

I5 *i5;

// expected-error@+1 2 {{'annotate' attribute takes at least 1 argument}}
#pragma clang attribute push (__attribute__((annotate)), apply_to=objc_implementation)
@interface I6 @end
@interface I6 (MyCat) @end
@interface I6 () @end

@implementation I6 @end // expected-note {{when applied to this declaration}}
@implementation I6 (MyCat) @end // expected-note {{when applied to this declaration}}

#pragma clang attribute pop
