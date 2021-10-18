// RUN: %clang_cc1 -std=gnu++98 -I %S/Inputs -triple x86_64-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -fexceptions -fobjc-exceptions -fcxx-exceptions -fobjc-arc-exceptions -O2 -disable-llvm-passes -o - %s | FileCheck %s

#include "literal-support.h"

struct X {
  X();
  ~X();
  operator id() const;
};

struct Y {
  Y();
  ~Y();
  operator id() const;
};

// CHECK-LABEL: define{{.*}} void @_Z10test_arrayv
void test_array() {
  // CHECK: [[ARR:%[a-zA-Z0-9.]+]] = alloca i8*
  // CHECK: [[OBJECTS:%[a-zA-Z0-9.]+]] = alloca [2 x i8*]
  // CHECK: [[TMPX:%[a-zA-Z0-9.]+]] = alloca %
  // CHECK: [[TMPY:%[a-zA-Z0-9.]+]] = alloca %

  // Initializing first element
  // CHECK: [[PTR1:%.*]] = bitcast i8** [[ARR]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 8, i8* [[PTR1]])
  // CHECK: [[ELEMENT0:%[a-zA-Z0-9.]+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[OBJECTS]], i64 0, i64 0
  // CHECK-NEXT: [[TMP_CAST:%.*]] = bitcast {{.*}} [[TMPX]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 1, i8* [[TMP_CAST]])
  // CHECK-NEXT: call void @_ZN1XC1Ev({{.*}} [[TMPX]])
  // CHECK-NEXT: [[OBJECT0:%[a-zA-Z0-9.]+]] = invoke i8* @_ZNK1XcvP11objc_objectEv
  // CHECK: [[RET0:%[a-zA-Z0-9.]+]] = notail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* [[OBJECT0]])
  // CHECK: store i8* [[RET0]], i8** [[ELEMENT0]]
  
  // Initializing the second element
  // CHECK: [[ELEMENT1:%[a-zA-Z0-9.]+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[OBJECTS]], i64 0, i64 1
  // CHECK-NEXT: [[TMP_CAST:%.*]] = bitcast {{.*}} [[TMPY]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 1, i8* [[TMP_CAST]])
  // CHECK-NEXT: invoke void @_ZN1YC1Ev({{.*}} [[TMPY]])
  // CHECK: [[OBJECT1:%[a-zA-Z0-9.]+]] = invoke i8* @_ZNK1YcvP11objc_objectEv
  // CHECK: [[RET1:%[a-zA-Z0-9.]+]] = notail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* [[OBJECT1]])
  // CHECK: store i8* [[RET1]], i8** [[ELEMENT1]]

  // Build the array
  // CHECK: {{invoke.*@objc_msgSend}}
  // CHECK: call i8* @llvm.objc.retainAutoreleasedReturnValue
  id arr = @[ X(), Y() ];

  // Destroy temporaries
  // CHECK-NOT: ret void
  // CHECK: call void @llvm.objc.release
  // CHECK-NOT: ret void
  // CHECK: invoke void @_ZN1YD1Ev
  // CHECK-NOT: ret void
  // CHECK: call void @llvm.objc.release
  // CHECK-NEXT: call void @_ZN1XD1Ev
  // CHECK-NOT: ret void
  // CHECK: call void @llvm.objc.release
  // CHECK-NEXT: [[PTR2:%.*]] = bitcast i8** [[ARR]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 8, i8* [[PTR2]])
  // CHECK-NEXT: ret void

  // Check cleanups
  // CHECK: call void @llvm.objc.release
  // CHECK-NOT: call void @llvm.objc.release
  // CHECK: invoke void @_ZN1YD1Ev
  // CHECK: call void @llvm.objc.release
  // CHECK-NOT: call void @llvm.objc.release
  // CHECK: invoke void @_ZN1XD1Ev
  // CHECK-NOT: call void @llvm.objc.release
  // CHECK: unreachable
}

// CHECK-LABEL: define weak_odr void @_Z24test_array_instantiationIiEvv
template<typename T>
void test_array_instantiation() {
  // CHECK: [[ARR:%[a-zA-Z0-9.]+]] = alloca i8*
  // CHECK: [[OBJECTS:%[a-zA-Z0-9.]+]] = alloca [2 x i8*]

  // Initializing first element
  // CHECK:      [[PTR1:%.*]] = bitcast i8** [[ARR]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.start.p0i8(i64 8, i8* [[PTR1]])
  // CHECK: [[ELEMENT0:%[a-zA-Z0-9.]+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[OBJECTS]], i64 0, i64 0
  // CHECK: call void @_ZN1XC1Ev
  // CHECK-NEXT: [[OBJECT0:%[a-zA-Z0-9.]+]] = invoke i8* @_ZNK1XcvP11objc_objectEv
  // CHECK: [[RET0:%[a-zA-Z0-9.]+]] = notail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* [[OBJECT0]])
  // CHECK: store i8* [[RET0]], i8** [[ELEMENT0]]
  
  // Initializing the second element
  // CHECK: [[ELEMENT1:%[a-zA-Z0-9.]+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[OBJECTS]], i64 0, i64 1
  // CHECK: invoke void @_ZN1YC1Ev
  // CHECK: [[OBJECT1:%[a-zA-Z0-9.]+]] = invoke i8* @_ZNK1YcvP11objc_objectEv
  // CHECK: [[RET1:%[a-zA-Z0-9.]+]] = notail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* [[OBJECT1]])
  // CHECK: store i8* [[RET1]], i8** [[ELEMENT1]]

  // Build the array
  // CHECK: {{invoke.*@objc_msgSend}}
  // CHECK: call i8* @llvm.objc.retainAutoreleasedReturnValue
  id arr = @[ X(), Y() ];

  // Destroy temporaries
  // CHECK-NOT: ret void
  // CHECK: call void @llvm.objc.release
  // CHECK-NOT: ret void
  // CHECK: invoke void @_ZN1YD1Ev
  // CHECK-NOT: ret void
  // CHECK: call void @llvm.objc.release
  // CHECK-NEXT: call void @_ZN1XD1Ev
  // CHECK-NOT: ret void
  // CHECK: call void @llvm.objc.release
  // CHECK-NEXT: [[PTR2]] = bitcast i8** [[ARR]] to i8*
  // CHECK-NEXT: call void @llvm.lifetime.end.p0i8(i64 8, i8* [[PTR2]])
  // CHECK-NEXT: ret void

  // Check cleanups
  // CHECK: call void @llvm.objc.release
  // CHECK-NOT: call void @llvm.objc.release
  // CHECK: invoke void @_ZN1YD1Ev
  // CHECK: call void @llvm.objc.release
  // CHECK-NOT: call void @llvm.objc.release
  // CHECK: invoke void @_ZN1XD1Ev
  // CHECK-NOT: call void @llvm.objc.release
  // CHECK: unreachable
}

template void test_array_instantiation<int>();

