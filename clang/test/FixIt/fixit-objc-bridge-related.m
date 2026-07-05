// RUN: not %clang_cc1 -triple x86_64-apple-darwin10  -fdiagnostics-parseable-fixits -x objective-c -fobjc-arc %s 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -triple x86_64-apple-darwin10  -fdiagnostics-parseable-fixits -x objective-c++ -fobjc-arc %s 2>&1 | FileCheck %s

typedef struct __attribute__((objc_bridge_related(UIColor,colorWithCGColor:,CGColor))) CGColor *CGColorRef;

@interface UIColor 
+ (UIColor *)colorWithCGColor:(CGColorRef)cgColor;
- (CGColorRef)CGColor;
@end

@interface UIButton
@property(nonatomic,retain) UIColor *tintColor;
@end

void test(UIButton *myButton) {
  CGColorRef cgColor = (CGColorRef)myButton.tintColor;
  cgColor = myButton.tintColor;

  cgColor = (CGColorRef)[myButton.tintColor CGColor];

  cgColor = (CGColorRef)[myButton tintColor];
}

// CHECK: {16:36-16:36}:"["
// CHECK: {16:54-16:54}:" CGColor]"

// CHECK: {17:13-17:13}:"["
// CHECK: {17:31-17:31}:" CGColor]"

// CHECK: {21:25-21:25}:"["
// CHECK: {21:45-21:45}:" CGColor]"

@interface ImplicitPropertyTest
- (UIColor *)tintColor;
@end

void test1(ImplicitPropertyTest *myImplicitPropertyTest) {
  CGColorRef cgColor = (CGColorRef)[myImplicitPropertyTest tintColor];
}

// CHECK: {38:36-38:36}:"["
// CHECK: {38:70-38:70}:" CGColor]"
