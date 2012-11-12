// RUN: %clang_cc1 -masm-verbose -S -g %s -o - | FileCheck %s

// CHECK: AT_APPLE_property_name
@interface C {
  int _base;
}
@property int base;
@end

@implementation C
@synthesize base = _base;
@end

void foo(C *cptr) {}
