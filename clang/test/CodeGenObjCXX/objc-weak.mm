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
// CHECK-LABEL: define linkonce_odr dereferenceable({{[0-9]+}}) %struct.A* @_ZN1AaSERKS_(
// CHECK:       [[THISADDR:%this.*]] = alloca [[A:.*]]*
// CHECK:       [[OBJECTADDR:%.*]] = alloca [[A:.*]]*
// CHECK:       [[THIS:%this.*]] = load [[A]]*, [[A]]** [[THISADDR]]
// CHECK:       [[OBJECT:%.*]] = load [[A]]*, [[A]]** [[OBJECTADDR]]
// CHECK:       [[T0:%.*]] = getelementptr inbounds [[A]], [[A]]* [[OBJECT]], i32 0, i32 0
// CHECK-NEXT:  [[T1:%.*]] = call i8* @llvm.objc.loadWeak(i8** [[T0]])
// CHECK-NEXT:  [[T2:%.*]] = getelementptr inbounds [[A]], [[A]]* [[THIS]], i32 0, i32 0
// CHECK-NEXT:  [[T3:%.*]] = call i8* @llvm.objc.storeWeak(i8** [[T2]], i8* [[T1]])

// Move Assignment Operator
// CHECK-LABEL: define linkonce_odr dereferenceable({{[0-9]+}}) %struct.A* @_ZN1AaSEOS_(
// CHECK:       [[THISADDR:%this.*]] = alloca [[A:.*]]*
// CHECK:       [[OBJECTADDR:%.*]] = alloca [[A:.*]]*
// CHECK:       [[THIS:%this.*]] = load [[A]]*, [[A]]** [[THISADDR]]
// CHECK:       [[OBJECT:%.*]] = load [[A]]*, [[A]]** [[OBJECTADDR]]
// CHECK:       [[T0:%.*]] = getelementptr inbounds [[A]], [[A]]* [[OBJECT]], i32 0, i32 0
// CHECK-NEXT:  [[T1:%.*]] = call i8* @llvm.objc.loadWeak(i8** [[T0]])
// CHECK-NEXT:  [[T2:%.*]] = getelementptr inbounds [[A]], [[A]]* [[THIS]], i32 0, i32 0
// CHECK-NEXT:  [[T3:%.*]] = call i8* @llvm.objc.storeWeak(i8** [[T2]], i8* [[T1]])

// Default Constructor
// CHECK-LABEL: define linkonce_odr void @_ZN1AC2Ev(
// CHECK:       [[THISADDR:%this.*]] = alloca [[A:.*]]*
// CHECK:       [[THIS:%this.*]] = load [[A]]*, [[A]]** [[THISADDR]]
// CHECK:       [[T0:%.*]] = getelementptr inbounds [[A]], [[A]]* [[THIS]], i32 0, i32 0
// CHECK-NEXT:  store i8* null, i8** [[T0]]

// Copy Constructor
// CHECK-LABEL: define linkonce_odr void @_ZN1AC2ERKS_(
// CHECK:       [[THISADDR:%this.*]] = alloca [[A:.*]]*
// CHECK:       [[OBJECTADDR:%.*]] = alloca [[A:.*]]*
// CHECK:       [[THIS:%this.*]] = load [[A]]*, [[A]]** [[THISADDR]]
// CHECK:       [[T0:%.*]] = getelementptr inbounds [[A]], [[A]]* [[THIS]], i32 0, i32 0
// CHECK-NEXT:  [[OBJECT:%.*]] = load [[A]]*, [[A]]** [[OBJECTADDR]]
// CHECK-NEXT:  [[T1:%.*]] = getelementptr inbounds [[A]], [[A]]* [[OBJECT]], i32 0, i32 0
// CHECK-NEXT:  call void @llvm.objc.copyWeak(i8** [[T0]], i8** [[T1]])

// Move Constructor
// CHECK-LABEL: define linkonce_odr void @_ZN1AC2EOS_(
// CHECK:       [[THISADDR:%this.*]] = alloca [[A:.*]]*
// CHECK:       [[OBJECTADDR:%.*]] = alloca [[A:.*]]*
// CHECK:       [[THIS:%this.*]] = load [[A]]*, [[A]]** [[THISADDR]]
// CHECK:       [[T0:%.*]] = getelementptr inbounds [[A]], [[A]]* [[THIS]], i32 0, i32 0
// CHECK-NEXT:  [[OBJECT:%.*]] = load [[A]]*, [[A]]** [[OBJECTADDR]]
// CHECK-NEXT:  [[T1:%.*]] = getelementptr inbounds [[A]], [[A]]* [[OBJECT]], i32 0, i32 0
// CHECK-NEXT:  call void @llvm.objc.moveWeak(i8** [[T0]], i8** [[T1]])

// Destructor
// CHECK-LABEL: define linkonce_odr void @_ZN1AD2Ev(
// CHECK:       [[THISADDR:%this.*]] = alloca [[A:.*]]*
// CHECK:       [[THIS:%this.*]] = load [[A]]*, [[A]]** [[THISADDR]]
// CHECK-NEXT:  [[T0:%.*]] = getelementptr inbounds [[A]], [[A]]* [[THIS]], i32 0, i32 0
// CHECK-NEXT:  call void @llvm.objc.destroyWeak(i8** [[T0]])

