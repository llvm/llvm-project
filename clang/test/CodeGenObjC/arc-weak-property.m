// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-runtime-has-weak -fblocks -fobjc-arc -o - %s | FileCheck %s

@interface WeakPropertyTest {
    __weak id PROP;
}
@property () __weak id PROP;
@end

@implementation WeakPropertyTest
@synthesize PROP;
@end

// CHECK:     define internal ptr @"\01-[WeakPropertyTest PROP]"
// CHECK:       [[SELF:%.*]] = alloca ptr,
// CHECK-NEXT:  [[CMD:%.*]] = alloca ptr,
// CHECK-NEXT:  store ptr {{%.*}}, ptr [[SELF]]
// CHECK-NEXT:  store ptr {{%.*}}, ptr [[CMD]]
// CHECK-NEXT:  [[T0:%.*]] = load ptr, ptr [[SELF]]
// CHECK-NEXT:  [[T1:%.*]] = load i64, ptr @"OBJC_IVAR_$_WeakPropertyTest.PROP"
// CHECK-NEXT:  [[T3:%.*]] = getelementptr inbounds i8, ptr [[T0]], i64 [[T1]]
// CHECK-NEXT:  [[T5:%.*]] = call ptr @llvm.objc.loadWeakRetained(ptr [[T3]])
// CHECK-NEXT:  [[T6:%.*]] = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr [[T5]])
// CHECK-NEXT:  ret ptr [[T6]]

// CHECK:     define internal void @"\01-[WeakPropertyTest setPROP:]"
// CHECK:       [[SELF:%.*]] = alloca ptr,
// CHECK-NEXT:  [[CMD:%.*]] = alloca ptr,
// CHECK-NEXT:  [[PROP:%.*]] = alloca ptr,
// CHECK-NEXT:  store ptr {{%.*}}, ptr [[SELF]]
// CHECK-NEXT:  store ptr {{%.*}}, ptr [[CMD]]
// CHECK-NEXT:  store ptr {{%.*}}, ptr [[PROP]]
// CHECK-NEXT:  [[V:%.*]] = load ptr, ptr [[PROP]]
// CHECK-NEXT:  [[T0:%.*]] = load ptr, ptr [[SELF]]
// CHECK-NEXT:  [[T1:%.*]] = load i64, ptr @"OBJC_IVAR_$_WeakPropertyTest.PROP"
// CHECK-NEXT:  [[T3:%.*]] = getelementptr inbounds i8, ptr [[T0]], i64 [[T1]]
// CHECK-NEXT:  call ptr @llvm.objc.storeWeak(ptr [[T3]], ptr [[V]])
// CHECK-NEXT:  ret void

// CHECK:     define internal void @"\01-[WeakPropertyTest .cxx_destruct]"
// CHECK:       [[SELF:%.*]] = alloca ptr,
// CHECK-NEXT:  [[CMD:%.*]] = alloca ptr,
// CHECK-NEXT:  store ptr {{%.*}}, ptr [[SELF]]
// CHECK-NEXT:  store ptr {{%.*}}, ptr [[CMD]]
// CHECK-NEXT:  [[T0:%.*]] = load ptr, ptr [[SELF]]
// CHECK-NEXT:  [[T1:%.*]] = load i64, ptr @"OBJC_IVAR_$_WeakPropertyTest.PROP"
// CHECK-NEXT:  [[T3:%.*]] = getelementptr inbounds i8, ptr [[T0]], i64 [[T1]]
// CHECK-NEXT:  call void @llvm.objc.destroyWeak(ptr [[T3]])
// CHECK-NEXT:  ret void
