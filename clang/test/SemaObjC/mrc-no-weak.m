// RUN: %clang_cc1 -fobjc-runtime=macosx-10.8 -fsyntax-only -verify %s

__attribute__((objc_root_class))
@interface Root @end

// These should not get diagnosed immediately.
@interface A : Root {
  __weak id x;
}
@property __weak id y;
@end

// Diagnostic goes on the ivar if it's explicit.
@interface B : Root {
  __weak id x;  // expected-error {{cannot create __weak reference in file using manual reference counting}}
}
@property __weak id x;
@end
@implementation B
@synthesize x;
@end

// Otherwise, it goes with the @synthesize.
@interface C : Root
@property __weak id x; // expected-note {{property declared here}}
@end
@implementation C
@synthesize x; // expected-error {{cannot synthesize weak property in file using manual reference counting}}
@end

@interface D : Root
@property __weak id x; // expected-note {{property declared here}}
@end
@implementation D // expected-error {{cannot synthesize weak property in file using manual reference counting}}
@end

@interface E : Root {
@public
  __weak id x; // expected-note 2 {{declaration uses __weak, but ARC is disabled}}
}
@end

void testE(E *e) {
  id x = e->x; // expected-error {{'x' is unavailable}}
  e->x = x; // expected-error {{'x' is unavailable}}
}

@interface F : Root
@property (weak) id x;
@end

void testF(F *f) {
  id x = f.x;
}
