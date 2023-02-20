// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fblocks -fobjc-weak -fobjc-runtime-has-weak -std=c++11 -o - %s | FileCheck %s

struct A { __weak id x; };

id test0() {
  A a;
  A b = a;
  A c(static_cast<A&&>(b));
  a = c;
  c = static_cast<A&&>(a);
  return c.x;
}

// Copy Assignment Operator
// CHECK-LABEL: define linkonce_odr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_ZN1AaSERKS_(
// CHECK:       [[THISADDR:%this.*]] = alloca ptr
// CHECK:       [[OBJECTADDR:%.*]] = alloca ptr
// CHECK:       [[THIS:%this.*]] = load ptr, ptr [[THISADDR]]
// CHECK:       [[OBJECT:%.*]] = load ptr, ptr [[OBJECTADDR]]
// CHECK:       [[T0:%.*]] = getelementptr inbounds [[A:.*]], ptr [[OBJECT]], i32 0, i32 0
// CHECK-NEXT:  [[T1:%.*]] = call ptr @llvm.objc.loadWeak(ptr [[T0]])
// CHECK-NEXT:  [[T2:%.*]] = getelementptr inbounds [[A:.*]], ptr [[THIS]], i32 0, i32 0
// CHECK-NEXT:  [[T3:%.*]] = call ptr @llvm.objc.storeWeak(ptr [[T2]], ptr [[T1]])

// Move Assignment Operator
// CHECK-LABEL: define linkonce_odr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_ZN1AaSEOS_(
// CHECK:       [[THISADDR:%this.*]] = alloca ptr
// CHECK:       [[OBJECTADDR:%.*]] = alloca ptr
// CHECK:       [[THIS:%this.*]] = load ptr, ptr [[THISADDR]]
// CHECK:       [[OBJECT:%.*]] = load ptr, ptr [[OBJECTADDR]]
// CHECK:       [[T0:%.*]] = getelementptr inbounds [[A:.*]], ptr [[OBJECT]], i32 0, i32 0
// CHECK-NEXT:  [[T1:%.*]] = call ptr @llvm.objc.loadWeak(ptr [[T0]])
// CHECK-NEXT:  [[T2:%.*]] = getelementptr inbounds [[A:.*]], ptr [[THIS]], i32 0, i32 0
// CHECK-NEXT:  [[T3:%.*]] = call ptr @llvm.objc.storeWeak(ptr [[T2]], ptr [[T1]])

// Default Constructor
// CHECK-LABEL: define linkonce_odr void @_ZN1AC2Ev(
// CHECK:       [[THISADDR:%this.*]] = alloca ptr
// CHECK:       [[THIS:%this.*]] = load ptr, ptr [[THISADDR]]
// CHECK:       [[T0:%.*]] = getelementptr inbounds [[A:.*]], ptr [[THIS]], i32 0, i32 0
// CHECK-NEXT:  store ptr null, ptr [[T0]]

// Copy Constructor
// CHECK-LABEL: define linkonce_odr void @_ZN1AC2ERKS_(
// CHECK:       [[THISADDR:%this.*]] = alloca ptr
// CHECK:       [[OBJECTADDR:%.*]] = alloca ptr
// CHECK:       [[THIS:%this.*]] = load ptr, ptr [[THISADDR]]
// CHECK:       [[T0:%.*]] = getelementptr inbounds [[A:.*]], ptr [[THIS]], i32 0, i32 0
// CHECK-NEXT:  [[OBJECT:%.*]] = load ptr, ptr [[OBJECTADDR]]
// CHECK-NEXT:  [[T1:%.*]] = getelementptr inbounds [[A:.*]], ptr [[OBJECT]], i32 0, i32 0
// CHECK-NEXT:  call void @llvm.objc.copyWeak(ptr [[T0]], ptr [[T1]])

// Move Constructor
// CHECK-LABEL: define linkonce_odr void @_ZN1AC2EOS_(
// CHECK:       [[THISADDR:%this.*]] = alloca ptr
// CHECK:       [[OBJECTADDR:%.*]] = alloca ptr
// CHECK:       [[THIS:%this.*]] = load ptr, ptr [[THISADDR]]
// CHECK:       [[T0:%.*]] = getelementptr inbounds [[A:.*]], ptr [[THIS]], i32 0, i32 0
// CHECK-NEXT:  [[OBJECT:%.*]] = load ptr, ptr [[OBJECTADDR]]
// CHECK-NEXT:  [[T1:%.*]] = getelementptr inbounds [[A:.*]], ptr [[OBJECT]], i32 0, i32 0
// CHECK-NEXT:  call void @llvm.objc.moveWeak(ptr [[T0]], ptr [[T1]])

// Destructor
// CHECK-LABEL: define linkonce_odr void @_ZN1AD2Ev(
// CHECK:       [[THISADDR:%this.*]] = alloca ptr
// CHECK:       [[THIS:%this.*]] = load ptr, ptr [[THISADDR]]
// CHECK-NEXT:  [[T0:%.*]] = getelementptr inbounds [[A:.*]], ptr [[THIS]], i32 0, i32 0
// CHECK-NEXT:  call void @llvm.objc.destroyWeak(ptr [[T0]])

