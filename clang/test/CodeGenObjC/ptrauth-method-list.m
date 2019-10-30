// RUN: %clang_cc1 -fptrauth-calls -fobjc-arc -fblocks -fobjc-runtime=ios-7 -triple arm64-apple-ios -emit-llvm -o - %s | FileCheck %s

// CHECK: @"\01+[C pm1].ptrauth" = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (i8*, i8*)* @"\01+[C pm1]" to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ i32, i32, [2 x %struct._objc_method] }, { i32, i32, [2 x %struct._objc_method] }* @"_OBJC_$_CLASS_METHODS_C", i32 0, i32 2, i32 0, i32 2) to i64), i64 0 }, section "llvm.ptrauth"
// CHECK: @"\01+[C m1].ptrauth" = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (i8*, i8*)* @"\01+[C m1]" to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ i32, i32, [2 x %struct._objc_method] }, { i32, i32, [2 x %struct._objc_method] }* @"_OBJC_$_CLASS_METHODS_C", i32 0, i32 2, i32 1, i32 2) to i64), i64 0 }, section "llvm.ptrauth"
// CHECK: @"_OBJC_$_CLASS_METHODS_C" = internal global { i32, i32, [2 x %struct._objc_method] } { i32 24, i32 2, [2 x %struct._objc_method] [%struct._objc_method { i8* getelementptr inbounds ([4 x i8], [4 x i8]* @OBJC_METH_VAR_NAME{{.*}}, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @OBJC_METH_VAR_TYPE_, i32 0, i32 0), i8* bitcast ({ i8*, i32, i64, i64 }* @"\01+[C pm1].ptrauth" to i8*) }, %struct._objc_method { i8* getelementptr inbounds ([3 x i8], [3 x i8]* @OBJC_METH_VAR_NAME{{.*}}, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @OBJC_METH_VAR_TYPE_, i32 0, i32 0), i8* bitcast ({ i8*, i32, i64, i64 }* @"\01+[C m1].ptrauth" to i8*) }] }, section "__DATA, __objc_const"
// CHECK: "\01-[C pm0].ptrauth" = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%0*, i8*)* @"\01-[C pm0]" to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ i32, i32, [2 x %struct._objc_method] }, { i32, i32, [2 x %struct._objc_method] }* @"_OBJC_$_INSTANCE_METHODS_C", i32 0, i32 2, i32 0, i32 2) to i64), i64 0 }, section "llvm.ptrauth"
// CHECK: "\01-[C m0].ptrauth" = private constant { i8*, i32, i64, i64 } { i8* bitcast (void (%0*, i8*)* @"\01-[C m0]" to i8*), i32 0, i64 ptrtoint (i8** getelementptr inbounds ({ i32, i32, [2 x %struct._objc_method] }, { i32, i32, [2 x %struct._objc_method] }* @"_OBJC_$_INSTANCE_METHODS_C", i32 0, i32 2, i32 1, i32 2) to i64), i64 0 }, section "llvm.ptrauth"
// CHECK: @"_OBJC_$_INSTANCE_METHODS_C" = internal global { i32, i32, [2 x %struct._objc_method] } { i32 24, i32 2, [2 x %struct._objc_method] [%struct._objc_method { i8* getelementptr inbounds ([4 x i8], [4 x i8]* @OBJC_METH_VAR_NAME{{.*}}, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @OBJC_METH_VAR_TYPE_, i32 0, i32 0), i8* bitcast ({ i8*, i32, i64, i64 }* @"\01-[C pm0].ptrauth" to i8*) }, %struct._objc_method { i8* getelementptr inbounds ([3 x i8], [3 x i8]* @OBJC_METH_VAR_NAME{{.*}}, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @OBJC_METH_VAR_TYPE_, i32 0, i32 0), i8* bitcast ({ i8*, i32, i64, i64 }* @"\01-[C m0].ptrauth" to i8*) }] }, section "__DATA, __objc_const"

@protocol P
- (void) pm0;
+ (void) pm1;
@end

@interface C<P>
- (void) m0;
+ (void) m1;
@end

@implementation C
- (void) pm0 {}
+ (void) pm1 {}
- (void) m0 {}
+ (void) m1 {}
@end

void test_method_list(C *c) {
  [c m0];
  [C m1];
}
