// Objective-C recovery
// RUN: not %clang_cc1  -triple x86_64-apple-darwin10  -fdiagnostics-parseable-fixits -x objective-c %s 2>&1  | FileCheck %s
// RUN: not %clang_cc1  -triple x86_64-apple-darwin10  -fobjc-arc -fdiagnostics-parseable-fixits -x objective-c %s 2>&1  | FileCheck %s
// RUN: not %clang_cc1  -triple x86_64-apple-darwin10  -fdiagnostics-parseable-fixits -x objective-c++ %s 2>&1  | FileCheck %s

typedef struct __attribute__((objc_bridge_related(NSColor,colorWithCGColor:,CGColor))) CGColor *CGColorRef;

@interface NSColor
+ (NSColor *)colorWithCGColor:(CGColorRef)cgColor;
- (CGColorRef)CGColor;
@end

@interface NSTextField
- (void)setBackgroundColor:(NSColor *)color;
- (NSColor *)backgroundColor;
@end

NSColor * Test1(NSTextField *textField, CGColorRef newColor) {
 textField.backgroundColor = newColor;
 return newColor;
}

CGColorRef Test2(NSTextField *textField, CGColorRef newColor) {
 newColor = textField.backgroundColor; // [textField.backgroundColor CGColor]
 return textField.backgroundColor;
}
// CHECK: {19:30-19:30}:"[NSColor colorWithCGColor:"
// CHECK: {19:38-19:38}:"]"
// CHECK: {20:9-20:9}:"[NSColor colorWithCGColor:"
// CHECK: {20:17-20:17}:"]"
// CHECK: {24:13-24:13}:"["
// CHECK: {24:38-24:38}:" CGColor]"
// CHECK: {25:9-25:9}:"["
// CHECK: {25:34-25:34}:" CGColor]"
