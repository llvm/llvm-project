// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fblocks -emit-llvm -o - %s | FileCheck %s

// CHECK: @{{.*}} = private global %{{.*}} { ptr @__CFConstantStringClassReference, i32 1992, ptr @{{.*}}, i64 3 }, section "__DATA,__cfstring", align 8 #[[ATTRNUM0:[0-9]+]]
// CHECK: @{{.*}} = internal constant { ptr, i32, i32, ptr, ptr } { ptr @_NSConcreteGlobalBlock, i32 1342177280, i32 0, ptr @{{.*}}, ptr @{{.*}} }, align 8 #[[ATTRNUM0]]

@class NSString;

NSString *testStringLiteral(void) {
  return @"abc";
}

int testGlobalBlock(int a) {
  return ^{ return 123; }();
}

// CHECK: attributes #[[ATTRNUM0]] = { "objc_arc_inert" }
