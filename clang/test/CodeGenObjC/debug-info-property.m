// RUN: %clang_cc1 -masm-verbose -S -g %s -o - | FileCheck %s

// CHECK: AT_APPLE_property_name
// CHECK: AT_APPLE_property_attribute
// CHECK: AT_APPLE_property
@interface I1 {
int p1;
}
@property int p1;
@end

@implementation I1
@synthesize p1;
@end
