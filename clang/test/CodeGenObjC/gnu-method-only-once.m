// RUN: %clang_cc1 -triple x86_64-unknown-freebsd -S -emit-llvm -fobjc-runtime=gnustep-2.0 -o - %s | FileCheck %s -check-prefix=CHECK-NEW
// RUN: %clang_cc1 -triple x86_64-unknown-freebsd -S -emit-llvm -fobjc-runtime=gnustep-1.8 -o - %s | FileCheck %s -check-prefix=CHECK-OLD

// Clang 9 or 10 changed the handling of method lists so that methods provided
// from synthesised properties showed up in the method list, where previously
// CGObjCGNU had to collect them and merge them.  One of the places where this
// merging happened was missed in the move and so we ended up emitting two
// copies of method metadata for declared properties.

// This class has only instance properties and only one pair of synthesized
// methods from the property and so we should synthesize only one method list,
// with precisely two methods on it.
@interface X
@property (retain) id iProp;
@end

@implementation X
@synthesize iProp;
@end

// Check that the method list has precisely 2 methods.
// CHECK-NEW: @.objc_method_list = internal global { ptr, i32, i64, [2 x
// CHECK-OLD: @.objc_method_list = internal global { ptr, i32, [2 x
