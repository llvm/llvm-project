// RUN: not %clang_cc1 -triple x86_64-apple-darwin10 -fblocks -fdiagnostics-parseable-fixits -x objective-c %s 2>&1 | FileCheck %s
// rdar://11311333

@interface NSView @end

@interface INTF
- (void) drawRect : inView:(NSView)view;
- (void)test:(NSView )a;
- (void)foo;
@end

// CHECK: {7:35-7:35}:"*"
// CHECK: {8:21-8:21}:"*"
@implementation INTF
-(void)foo {
  ^(NSView view) {
  };
}
@end
// CHECK: {16:11-16:11}:"*"
