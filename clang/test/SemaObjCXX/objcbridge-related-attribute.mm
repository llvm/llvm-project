// RUN: %clang_cc1 -fsyntax-only -x objective-c++ -verify -Wno-objc-root-class %s
// rdar://15499111
typedef struct __attribute__((objc_bridge_related(NSColor,colorWithCGColor:,CGColor))) CGColor *CGColorRef; // expected-note 6 {{declared here}}

@interface NSColor // expected-note 6 {{declared here}}
+ (NSColor *)colorWithCGColor:(CGColorRef)cgColor;
- (CGColorRef)CGColor;
@end

@interface NSTextField
- (void)setBackgroundColor:(NSColor *)color;
- (NSColor *)backgroundColor;
@end


NSColor *Test1(NSColor *nsColor, CGColorRef newColor) {
  nsColor = newColor; // expected-error {{'CGColorRef' (aka 'CGColor *') must be explicitly converted to 'NSColor *'; use '+colorWithCGColor:' method for this conversion}}
  NSColor *ns = newColor; // expected-error {{'CGColorRef' (aka 'CGColor *') must be explicitly converted to 'NSColor *'; use '+colorWithCGColor:' method for this conversion}} 
  return newColor; // expected-error {{'CGColorRef' (aka 'CGColor *') must be explicitly converted to 'NSColor *'; use '+colorWithCGColor:' method for this conversion}} 
}

CGColorRef Test2(NSColor *newColor, CGColorRef cgColor) {
  cgColor = newColor; // expected-error {{'NSColor *' must be explicitly converted to 'CGColorRef' (aka 'CGColor *'); use '-CGColor' method for this conversion}}
  CGColorRef cg = newColor; // expected-error {{'NSColor *' must be explicitly converted to 'CGColorRef' (aka 'CGColor *'); use '-CGColor' method for this conversion}} 
  return newColor; // expected-error {{'NSColor *' must be explicitly converted to 'CGColorRef' (aka 'CGColor *'); use '-CGColor' method for this conversion}}
}

