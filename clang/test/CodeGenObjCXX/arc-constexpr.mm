// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-arc -fobjc-runtime-has-weak -o - -std=c++11 %s | FileCheck %s

// CHECK: @[[CFSTRING:[a-z0-9_]+]] = private global %struct.__NSConstantString_tag
@class NSString;

// CHECK-LABEL: define{{.*}} void @_Z5test1v
// CHECK:   %[[ALLOCA:[A-Z]+]] = alloca ptr
// CHECK:   %[[V0:[0-9]+]] = call ptr @llvm.objc.retain(ptr @[[CFSTRING]]
// CHECK:   store ptr %[[V0]], ptr %[[ALLOCA]]
// CHECK:   call void @llvm.objc.storeStrong(ptr %[[ALLOCA]], ptr null)
void test1() {
  constexpr NSString *S = @"abc";
}

// CHECK-LABEL: define{{.*}} void @_Z5test2v
// CHECK:      %[[CONST:[a-zA-Z]+]] = alloca ptr
// CHECK:      %[[REF_CONST:[a-zA-Z]+]] = alloca ptr
// CHECK:      %[[V0:[0-9]+]] = call ptr @llvm.objc.retain(ptr @[[CFSTRING]]
// CHECK-NEXT: store ptr %[[V0]], ptr %[[CONST]]
// CHECK:      %[[V2:[0-9]+]] = call ptr @llvm.objc.retain(ptr @[[CFSTRING]]
// CHECK-NEXT: store ptr %[[V2]], ptr %[[REF_CONST]]
// CHECK-NEXT: call void @llvm.objc.storeStrong(ptr %[[REF_CONST]], ptr null)
// CHECK-NEXT: call void @llvm.objc.storeStrong(ptr %[[CONST]], ptr null)
void test2() {
  constexpr NSString *Const = @"abc";
  // In IR RefConst should be initialized with Const initializer instead of
  // reading from variable.
  NSString* RefConst = Const;
}

// CHECK-LABEL: define{{.*}} void @_Z5test3v
// CHECK:      %[[WEAK_CONST:[a-zA-Z]+]] = alloca ptr
// CHECK:      %[[REF_WEAK_CONST:[a-zA-Z]+]] = alloca ptr
// CHECK-NEXT: %[[V1:[0-9]+]] = call ptr @llvm.objc.initWeak(ptr %[[WEAK_CONST]], ptr @[[CFSTRING]]
// CHECK:      store ptr @[[CFSTRING]], ptr %[[REF_WEAK_CONST]]
// CHECK-NEXT: call void @llvm.objc.storeStrong(ptr %[[REF_WEAK_CONST]], ptr null)
// CHECK-NEXT: call void @llvm.objc.destroyWeak(ptr %[[WEAK_CONST]])
void test3() {
  __weak constexpr NSString *WeakConst = @"abc";
  NSString* RefWeakConst = WeakConst;
}
