// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin10 -emit-llvm -fobjc-arc -fobjc-runtime-has-weak -o - -std=c++11 %s | FileCheck %s

// CHECK: %[[TYPE:[a-z0-9]+]] = type opaque
// CHECK: @[[CFSTRING:[a-z0-9_]+]] = private global %struct.__NSConstantString_tag
@class NSString;

// CHECK-LABEL: define{{.*}} void @_Z5test1v
// CHECK:   %[[ALLOCA:[A-Z]+]] = alloca %[[TYPE]]*
// CHECK:   %[[V0:[0-9]+]] = call i8* @llvm.objc.retain(i8* bitcast (%struct.__NSConstantString_tag* @[[CFSTRING]]
// CHECK:   %[[V1:[0-9]+]] = bitcast i8* %[[V0]] to %[[TYPE]]*
// CHECK:   store %[[TYPE]]* %[[V1]], %[[TYPE]]** %[[ALLOCA]]
// CHECK:   %[[V2:[0-9]+]] = bitcast %[[TYPE]]** %[[ALLOCA]]
// CHECK:   call void @llvm.objc.storeStrong(i8** %[[V2]], i8* null)
void test1() {
  constexpr NSString *S = @"abc";
}

// CHECK-LABEL: define{{.*}} void @_Z5test2v
// CHECK:      %[[CONST:[a-zA-Z]+]] = alloca %[[TYPE]]*
// CHECK:      %[[REF_CONST:[a-zA-Z]+]] = alloca %[[TYPE]]*
// CHECK:      %[[V0:[0-9]+]] = call i8* @llvm.objc.retain(i8* bitcast (%struct.__NSConstantString_tag* @[[CFSTRING]]
// CHECK-NEXT: %[[V1:[0-9]+]] = bitcast i8* %[[V0]] to %[[TYPE]]*
// CHECK-NEXT: store %[[TYPE]]* %[[V1]], %[[TYPE]]** %[[CONST]]
// CHECK:      %[[V2:[0-9]+]] = call i8* @llvm.objc.retain(i8* bitcast (%struct.__NSConstantString_tag* @[[CFSTRING]]
// CHECK-NEXT: %[[V3:[0-9]+]] = bitcast i8* %[[V2]] to %[[TYPE]]*
// CHECK-NEXT: store %[[TYPE]]* %[[V3]], %[[TYPE]]** %[[REF_CONST]]
// CHECK:      %[[V4:[0-9]+]] = bitcast %[[TYPE]]** %[[REF_CONST]]
// CHECK-NEXT: call void @llvm.objc.storeStrong(i8** %[[V4]], i8* null)
// CHECK:      %[[V5:[0-9]+]] = bitcast %[[TYPE]]** %[[CONST]]
// CHECK-NEXT: call void @llvm.objc.storeStrong(i8** %[[V5]], i8* null)
void test2() {
  constexpr NSString *Const = @"abc";
  // In IR RefConst should be initialized with Const initializer instead of
  // reading from variable.
  NSString* RefConst = Const;
}

// CHECK-LABEL: define{{.*}} void @_Z5test3v
// CHECK:      %[[WEAK_CONST:[a-zA-Z]+]] = alloca %[[TYPE]]*
// CHECK:      %[[REF_WEAK_CONST:[a-zA-Z]+]] = alloca %[[TYPE]]*
// CHECK:      %[[V0:[0-9]+]] = bitcast %[[TYPE]]** %[[WEAK_CONST]]
// CHECK-NEXT: %[[V1:[0-9]+]] = call i8* @llvm.objc.initWeak(i8** %[[V0]], i8* bitcast (%struct.__NSConstantString_tag* @[[CFSTRING]]
// CHECK:      store %[[TYPE]]* bitcast (%struct.__NSConstantString_tag* @[[CFSTRING]] to %[[TYPE]]*), %[[TYPE]]** %[[REF_WEAK_CONST]]
// CHECK:      %[[V2:[0-9]+]] = bitcast %[[TYPE]]** %[[REF_WEAK_CONST]]
// CHECK-NEXT: call void @llvm.objc.storeStrong(i8** %[[V2]], i8* null)
// CHECK:      %[[V3:[0-9]+]] = bitcast %[[TYPE]]** %[[WEAK_CONST]]
// CHECK-NEXT: call void @llvm.objc.destroyWeak(i8** %[[V3]])
void test3() {
  __weak constexpr NSString *WeakConst = @"abc";
  NSString* RefWeakConst = WeakConst;
}
