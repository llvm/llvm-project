// Objective-C recovery
// RUN: not %clang_cc1  -triple x86_64-apple-darwin10  -fdiagnostics-parseable-fixits -x objective-c %s 2>&1  | FileCheck %s
// RUN: not %clang_cc1  -triple x86_64-apple-darwin10  -fobjc-arc -fdiagnostics-parseable-fixits -x objective-c %s 2>&1  | FileCheck %s
// RUN: not %clang_cc1  -triple x86_64-apple-darwin10  -fdiagnostics-parseable-fixits -x objective-c++ %s 2>&1  | FileCheck %s
// rdar://15499111

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
// CHECK: {20:30-20:30}:"[NSColor colorWithCGColor:"
// CHECK: {20:38-20:38}:"]"
// CHECK: {21:9-21:9}:"[NSColor colorWithCGColor:"
// CHECK: {21:17-21:17}:"]"
// CHECK: {25:13-25:13}:"["
// CHECK: {25:38-25:38}:" CGColor]"
// CHECK: {26:9-26:9}:"["
// CHECK: {26:34-26:34}:" CGColor]"
