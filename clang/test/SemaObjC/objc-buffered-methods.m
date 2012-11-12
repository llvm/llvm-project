// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
// expected-no-diagnostics
// rdar://8843851

int* global;

@interface I
- (void) Meth;
@property int prop;
@property int prop1;
@end

@implementation I
+ (void) _defaultMinSize { };
static void _initCommon() {
  Class graphicClass;
  [graphicClass _defaultMinSize];
}

- (void) Meth { [self Forw]; } // No warning now
- (void) Forw {}
- (int) func { return prop; }  // compiles - synthesized ivar will be accessible here.
- (int)get_g { return global; } // No warning here - synthesized ivar will be accessible here.
@synthesize prop;
@synthesize prop1=global;
@end
