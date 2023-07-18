// RUN: not %clang_cc1  -triple x86_64-apple-darwin10  -fdiagnostics-parseable-fixits -x objective-c %s 2>&1  | FileCheck %s
// RUN: not %clang_cc1  -triple x86_64-apple-darwin10  -fobjc-arc -fdiagnostics-parseable-fixits -x objective-c %s 2>&1  | FileCheck %s
// RUN: not %clang_cc1  -triple x86_64-apple-darwin10  -fdiagnostics-parseable-fixits -x objective-c++ %s 2>&1  | FileCheck %s
// rdar://15517899

typedef struct __attribute__((objc_bridge_related(NSColor,colorWithCGColor:,CGColor))) CGColor *CGColorRef;

@interface NSColor
+ (NSColor *)colorWithCGColor:(CGColorRef)cgColor;
@property CGColorRef CGColor;
@end

@interface NSTextField
- (void)setBackgroundColor:(NSColor *)color;
- (NSColor *)backgroundColor;
@end

CGColorRef Test(NSTextField *textField, CGColorRef newColor) {
 newColor = textField.backgroundColor;
 return textField.backgroundColor;
}
// CHECK:{19:38-19:38}:".CGColor"
// CHECK:{20:34-20:34}:".CGColor"
