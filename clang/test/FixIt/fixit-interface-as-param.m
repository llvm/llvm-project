// RUN: not %clang_cc1 -triple x86_64-apple-darwin10 -fblocks -fdiagnostics-parseable-fixits -x objective-c %s 2>&1 | FileCheck %s

@interface NSView @end

@interface INTF
- (void) drawRect : inView:(NSView)view;
- (void)test:(NSView )a;
- (void)foo;
@end

// CHECK: {6:35-6:35}:"*"
// CHECK: {7:21-7:21}:"*"
@implementation INTF
-(void)foo {
  ^(NSView view) {
  };
}
@end
// CHECK: {15:11-15:11}:"*"
