// RUN: %clang_cc1 -fsyntax-only -x objective-c -verify -Wno-objc-root-class %s

typedef struct __attribute__((objc_bridge_related(NSColor,colorXWithCGColor:,CXGColor))) CGColor *CGColorRef; // expected-note 2 {{declared here}}

typedef struct __attribute__((objc_bridge_related(XNSColor,colorWithCGColor:,CGColor))) CGColor1 *CGColorRef1; // expected-note 2 {{declared here}}

typedef struct __attribute__((objc_bridge_related(PNsColor,colorWithCGColor:,CGColor))) CGColor2 *CGColorRef2; // expected-note 2 {{declared here}}

@interface NSColor
+ (NSColor *)colorWithCGColor:(CGColorRef)cgColor;
- (CGColorRef)CGColor;
@end

@interface NSTextField 
- (void)setBackgroundColor:(NSColor *)color;
- (NSColor *)backgroundColor;
@end

typedef int PNsColor; // expected-note 2 {{declared here}}

NSColor * Test1(NSTextField *textField, CGColorRef newColor) {
 textField.backgroundColor = newColor; // expected-error {{'CGColorRef' (aka 'struct CGColor *') must be explicitly converted to 'NSColor *'; use '+colorXWithCGColor:' method for this conversion}} \
					// expected-warning {{incompatible pointer types assigning to 'NSColor *' from 'CGColorRef' (aka 'struct CGColor *')}}
 newColor = textField.backgroundColor; // expected-error {{'NSColor *' must be explicitly converted to 'CGColorRef' (aka 'struct CGColor *'); use '-CXGColor' method for this conversion}} \
					// expected-warning {{incompatible pointer types assigning to 'CGColorRef' (aka 'struct CGColor *') from 'NSColor *'}}
}
NSColor * Test2(NSTextField *textField, CGColorRef1 newColor) {
 textField.backgroundColor = newColor; // expected-error {{could not find Objective-C class 'XNSColor' to convert 'CGColorRef1' (aka 'struct CGColor1 *') to 'NSColor *'}} \
				       // expected-warning {{incompatible pointer types assigning to 'NSColor *' from 'CGColorRef1' (aka 'struct CGColor1 *')}}
 newColor = textField.backgroundColor ; // expected-error {{could not find Objective-C class 'XNSColor' to convert 'NSColor *' to 'CGColorRef1' (aka 'struct CGColor1 *')}} \
					// expected-warning {{incompatible pointer types assigning to 'CGColorRef1' (aka 'struct CGColor1 *') from 'NSColor *'}}
}

NSColor * Test3(NSTextField *textField, CGColorRef2 newColor) {
 textField.backgroundColor = newColor; // expected-error {{'PNsColor' must be name of an Objective-C class to be able to convert 'CGColorRef2' (aka 'struct CGColor2 *') to 'NSColor *'}} \
					// expected-warning {{incompatible pointer types assigning to 'NSColor *' from 'CGColorRef2' (aka 'struct CGColor2 *')}}
 newColor = textField.backgroundColor; // expected-error {{'PNsColor' must be name of an Objective-C class to be able to convert 'NSColor *' to 'CGColorRef2' (aka 'struct CGColor2 *')}} \
					// expected-warning {{incompatible pointer types assigning to 'CGColorRef2' (aka 'struct CGColor2 *') from 'NSColor *'}}
}

