// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fblocks -emit-llvm -o - %s | FileCheck %s

// CHECK: @{{.*}} = private global %{{.*}} { i32* getelementptr inbounds ([0 x i32], [0 x i32]* @__CFConstantStringClassReference, i32 0, i32 0), i32 1992, i8* getelementptr inbounds ([4 x i8], [4 x i8]* @{{.*}}, i32 0, i32 0), i64 3 }, section "__DATA,__cfstring", align 8 #[[ATTRNUM0:[0-9]+]]
// CHECK: @{{.*}} = internal constant { i8**, i32, i32, i8*, %{{.*}}* } { i8** @_NSConcreteGlobalBlock, i32 1342177280, i32 0, i8* bitcast (i32 (i8*)* @{{.*}} to i8*), %{{.*}}* bitcast ({ i64, i64, i8*, i8* }* @{{.*}} to %{{.*}}*) }, align 8 #[[ATTRNUM0]]

@class NSString;

NSString *testStringLiteral(void) {
  return @"abc";
}

int testGlobalBlock(int a) {
  return ^{ return 123; }();
}

// CHECK: attributes #[[ATTRNUM0]] = { "objc_arc_inert" }
