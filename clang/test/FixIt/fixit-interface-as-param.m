// RUN: not %clang_cc1 -triple x86_64-apple-darwin10  -fdiagnostics-parseable-fixits -x objective-c %s 2>&1 | FileCheck %s
// rdar://11311333

@interface NSView @end

@interface INTF
- (void) drawRect : inView:(NSView)view;
@end

// CHECK: {7:35-7:35}:"*"

