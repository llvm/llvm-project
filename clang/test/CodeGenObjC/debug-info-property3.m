// RUN: %clang_cc1 -S -emit-llvm -g %s -o - | FileCheck %s

// CHECK: metadata !"p1", metadata !6, i32 5, metadata !"", metadata !"", i32 2316, metadata !9} ; [ DW_TAG_APPLE_property ]
@interface I1
@property int p1;
@end

@implementation I1
@synthesize p1;
@end

void foo(I1 *iptr) {}
