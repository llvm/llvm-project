// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-runtime-has-weak -fblocks -fobjc-arc -o - %s | FileCheck %s
// rdar://8899430

@interface WeakPropertyTest {
    __weak id PROP;
}
@property () __weak id PROP;
@end

@implementation WeakPropertyTest
@synthesize PROP;
@end

// CHECK:     define internal i8* @"\01-[WeakPropertyTest PROP]"
// CHECK:       [[SELF:%.*]] = alloca [[WPT:%.*]]*,
// CHECK-NEXT:  [[CMD:%.*]] = alloca i8*,
// CHECK-NEXT:  store [[WPT]]* {{%.*}}, [[WPT]]** [[SELF]]
// CHECK-NEXT:  store i8* {{%.*}}, i8** [[CMD]]
// CHECK-NEXT:  [[T0:%.*]] = load [[WPT]]*, [[WPT]]** [[SELF]]
// CHECK-NEXT:  [[T1:%.*]] = load i64, i64* @"OBJC_IVAR_$_WeakPropertyTest.PROP"
// CHECK-NEXT:  [[T2:%.*]] = bitcast [[WPT]]* [[T0]] to i8*
// CHECK-NEXT:  [[T3:%.*]] = getelementptr inbounds i8, i8* [[T2]], i64 [[T1]]
// CHECK-NEXT:  [[T4:%.*]] = bitcast i8* [[T3]] to i8**
// CHECK-NEXT:  [[T5:%.*]] = call i8* @llvm.objc.loadWeakRetained(i8** [[T4]])
// CHECK-NEXT:  [[T6:%.*]] = tail call i8* @llvm.objc.autoreleaseReturnValue(i8* [[T5]])
// CHECK-NEXT:  ret i8* [[T6]]

// CHECK:     define internal void @"\01-[WeakPropertyTest setPROP:]"
// CHECK:       [[SELF:%.*]] = alloca [[WPT:%.*]]*,
// CHECK-NEXT:  [[CMD:%.*]] = alloca i8*,
// CHECK-NEXT:  [[PROP:%.*]] = alloca i8*,
// CHECK-NEXT:  store [[WPT]]* {{%.*}}, [[WPT]]** [[SELF]]
// CHECK-NEXT:  store i8* {{%.*}}, i8** [[CMD]]
// CHECK-NEXT:  store i8* {{%.*}}, i8** [[PROP]]
// CHECK-NEXT:  [[V:%.*]] = load i8*, i8** [[PROP]]
// CHECK-NEXT:  [[T0:%.*]] = load [[WPT]]*, [[WPT]]** [[SELF]]
// CHECK-NEXT:  [[T1:%.*]] = load i64, i64* @"OBJC_IVAR_$_WeakPropertyTest.PROP"
// CHECK-NEXT:  [[T2:%.*]] = bitcast [[WPT]]* [[T0]] to i8*
// CHECK-NEXT:  [[T3:%.*]] = getelementptr inbounds i8, i8* [[T2]], i64 [[T1]]
// CHECK-NEXT:  [[T4:%.*]] = bitcast i8* [[T3]] to i8**
// CHECK-NEXT:  call i8* @llvm.objc.storeWeak(i8** [[T4]], i8* [[V]])
// CHECK-NEXT:  ret void

// CHECK:     define internal void @"\01-[WeakPropertyTest .cxx_destruct]"
// CHECK:       [[SELF:%.*]] = alloca [[WPT:%.*]]*,
// CHECK-NEXT:  [[CMD:%.*]] = alloca i8*,
// CHECK-NEXT:  store [[WPT]]* {{%.*}}, [[WPT]]** [[SELF]]
// CHECK-NEXT:  store i8* {{%.*}}, i8** [[CMD]]
// CHECK-NEXT:  [[T0:%.*]] = load [[WPT]]*, [[WPT]]** [[SELF]]
// CHECK-NEXT:  [[T1:%.*]] = load i64, i64* @"OBJC_IVAR_$_WeakPropertyTest.PROP"
// CHECK-NEXT:  [[T2:%.*]] = bitcast [[WPT]]* [[T0]] to i8*
// CHECK-NEXT:  [[T3:%.*]] = getelementptr inbounds i8, i8* [[T2]], i64 [[T1]]
// CHECK-NEXT:  [[T4:%.*]] = bitcast i8* [[T3]] to i8**
// CHECK-NEXT:  call void @llvm.objc.destroyWeak(i8** [[T4]])
// CHECK-NEXT:  ret void
