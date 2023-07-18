// RUN: not %clang_cc1 -triple x86_64-apple-darwin10  -fdiagnostics-parseable-fixits -x objective-c -fobjc-arc %s 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -triple x86_64-apple-darwin10  -fdiagnostics-parseable-fixits -x objective-c++ -fobjc-arc %s 2>&1 | FileCheck %s
// rdar://15932435

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

// CHECK: {17:36-17:36}:"["
// CHECK: {17:54-17:54}:" CGColor]"

// CHECK: {18:13-18:13}:"["
// CHECK: {18:31-18:31}:" CGColor]"

// CHECK: {22:25-22:25}:"["
// CHECK: {22:45-22:45}:" CGColor]"

@interface ImplicitPropertyTest
- (UIColor *)tintColor;
@end

void test1(ImplicitPropertyTest *myImplicitPropertyTest) {
  CGColorRef cgColor = (CGColorRef)[myImplicitPropertyTest tintColor];
}

// CHECK: {39:36-39:36}:"["
// CHECK: {39:70-39:70}:" CGColor]"
