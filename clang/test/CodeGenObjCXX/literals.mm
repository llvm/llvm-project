// RUN: %clang_cc1 -I %S/Inputs -triple x86_64-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -fobjc-runtime-has-weak -fexceptions -fobjc-exceptions -fcxx-exceptions -fobjc-arc-exceptions -O2 -disable-llvm-optzns -o - %s | FileCheck %s

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

// CHECK: define void @_Z10test_arrayv
void test_array() {
  // CHECK: [[OBJECTS:%[a-zA-Z0-9.]+]] = alloca [2 x i8*]

  // Initializing first element
  // CHECK: [[ELEMENT0:%[a-zA-Z0-9.]+]] = getelementptr inbounds [2 x i8*]* [[OBJECTS]], i32 0, i32 0
  // CHECK-NEXT: call void @_ZN1XC1Ev
  // CHECK-NEXT: [[OBJECT0:%[a-zA-Z0-9.]+]] = invoke i8* @_ZNK1XcvP11objc_objectEv
  // CHECK: [[RET0:%[a-zA-Z0-9.]+]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[OBJECT0]])
  // CHECK: store i8* [[RET0]], i8** [[ELEMENT0]]
  
  // Initializing the second element
  // CHECK: [[ELEMENT1:%[a-zA-Z0-9.]+]] = getelementptr inbounds [2 x i8*]* [[OBJECTS]], i32 0, i32 1
  // CHECK-NEXT: invoke void @_ZN1YC1Ev
  // CHECK: [[OBJECT1:%[a-zA-Z0-9.]+]] = invoke i8* @_ZNK1YcvP11objc_objectEv
  // CHECK: [[RET1:%[a-zA-Z0-9.]+]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[OBJECT1]])
  // CHECK: store i8* [[RET1]], i8** [[ELEMENT1]]

  // Build the array
  // CHECK: {{invoke.*@objc_msgSend}}
  // CHECK: call i8* @objc_retainAutoreleasedReturnValue
  id arr = @[ X(), Y() ];

  // Destroy temporaries
  // CHECK-NOT: ret void
  // CHECK: call void @objc_release
  // CHECK-NOT: ret void
  // CHECK: invoke void @_ZN1YD1Ev
  // CHECK-NOT: ret void
  // CHECK: call void @objc_release
  // CHECK-NEXT: call void @_ZN1XD1Ev
  // CHECK-NOT: ret void
  // CHECK: call void @objc_release
  // CHECK-NEXT: ret void

  // Check cleanups
  // CHECK: call void @objc_release
  // CHECK-NOT: call void @objc_release
  // CHECK: invoke void @_ZN1YD1Ev
  // CHECK: call void @objc_release
  // CHECK-NOT: call void @objc_release
  // CHECK: invoke void @_ZN1XD1Ev
  // CHECK-NOT: call void @objc_release
  // CHECK: unreachable
}

// CHECK: define weak_odr void @_Z24test_array_instantiationIiEvv
template<typename T>
void test_array_instantiation() {
  // CHECK: [[OBJECTS:%[a-zA-Z0-9.]+]] = alloca [2 x i8*]

  // Initializing first element
  // CHECK: [[ELEMENT0:%[a-zA-Z0-9.]+]] = getelementptr inbounds [2 x i8*]* [[OBJECTS]], i32 0, i32 0
  // CHECK-NEXT: call void @_ZN1XC1Ev
  // CHECK-NEXT: [[OBJECT0:%[a-zA-Z0-9.]+]] = invoke i8* @_ZNK1XcvP11objc_objectEv
  // CHECK: [[RET0:%[a-zA-Z0-9.]+]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[OBJECT0]])
  // CHECK: store i8* [[RET0]], i8** [[ELEMENT0]]
  
  // Initializing the second element
  // CHECK: [[ELEMENT1:%[a-zA-Z0-9.]+]] = getelementptr inbounds [2 x i8*]* [[OBJECTS]], i32 0, i32 1
  // CHECK-NEXT: invoke void @_ZN1YC1Ev
  // CHECK: [[OBJECT1:%[a-zA-Z0-9.]+]] = invoke i8* @_ZNK1YcvP11objc_objectEv
  // CHECK: [[RET1:%[a-zA-Z0-9.]+]] = call i8* @objc_retainAutoreleasedReturnValue(i8* [[OBJECT1]])
  // CHECK: store i8* [[RET1]], i8** [[ELEMENT1]]

  // Build the array
  // CHECK: {{invoke.*@objc_msgSend}}
  // CHECK: call i8* @objc_retainAutoreleasedReturnValue
  id arr = @[ X(), Y() ];

  // Destroy temporaries
  // CHECK-NOT: ret void
  // CHECK: call void @objc_release
  // CHECK-NOT: ret void
  // CHECK: invoke void @_ZN1YD1Ev
  // CHECK-NOT: ret void
  // CHECK: call void @objc_release
  // CHECK-NEXT: call void @_ZN1XD1Ev
  // CHECK-NOT: ret void
  // CHECK: call void @objc_release
  // CHECK-NEXT: ret void

  // Check cleanups
  // CHECK: call void @objc_release
  // CHECK-NOT: call void @objc_release
  // CHECK: invoke void @_ZN1YD1Ev
  // CHECK: call void @objc_release
  // CHECK-NOT: call void @objc_release
  // CHECK: invoke void @_ZN1XD1Ev
  // CHECK-NOT: call void @objc_release
  // CHECK: unreachable
}

template void test_array_instantiation<int>();

