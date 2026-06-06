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
  // CHECK: [[ARR:%[a-zA-Z0-9.]+]] = alloca ptr
  // CHECK: [[OBJECTS:%[a-zA-Z0-9.]+]] = alloca [2 x ptr]
  // CHECK: [[TMPX:%[a-zA-Z0-9.]+]] = alloca %
  // CHECK: [[TMPY:%[a-zA-Z0-9.]+]] = alloca %

  // Initializing first element
  // CHECK: call void @llvm.lifetime.start.p0(ptr [[ARR]])
  // CHECK: [[ELEMENT0:%[a-zA-Z0-9.]+]] = getelementptr inbounds [2 x ptr], ptr [[OBJECTS]], i64 0, i64 0
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr [[TMPX]])
  // CHECK-NEXT: call void @_ZN1XC1Ev({{.*}} [[TMPX]])
  // CHECK-NEXT: [[OBJECT0:%[a-zA-Z0-9.]+]] = invoke noundef ptr @_ZNK1XcvP11objc_objectEv{{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK: store ptr [[OBJECT0]], ptr [[ELEMENT0]]
  
  // Initializing the second element
  // CHECK: [[ELEMENT1:%[a-zA-Z0-9.]+]] = getelementptr inbounds [2 x ptr], ptr [[OBJECTS]], i64 0, i64 1
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr [[TMPY]])
  // CHECK-NEXT: invoke void @_ZN1YC1Ev({{.*}} [[TMPY]])
  // CHECK: [[OBJECT1:%[a-zA-Z0-9.]+]] = invoke noundef ptr @_ZNK1YcvP11objc_objectEv{{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK: store ptr [[OBJECT1]], ptr [[ELEMENT1]]

  // Build the array
  // CHECK: {{invoke.*@objc_msgSend}}{{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
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
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr [[ARR]])
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
  // CHECK: [[ARR:%[a-zA-Z0-9.]+]] = alloca ptr
  // CHECK: [[OBJECTS:%[a-zA-Z0-9.]+]] = alloca [2 x ptr]

  // Initializing first element
  // CHECK: call void @llvm.lifetime.start.p0(ptr [[ARR]])
  // CHECK: [[ELEMENT0:%[a-zA-Z0-9.]+]] = getelementptr inbounds [2 x ptr], ptr [[OBJECTS]], i64 0, i64 0
  // CHECK: call void @_ZN1XC1Ev
  // CHECK-NEXT: [[OBJECT0:%[a-zA-Z0-9.]+]] = invoke noundef ptr @_ZNK1XcvP11objc_objectEv{{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK: store ptr [[OBJECT0]], ptr [[ELEMENT0]]
  
  // Initializing the second element
  // CHECK: [[ELEMENT1:%[a-zA-Z0-9.]+]] = getelementptr inbounds [2 x ptr], ptr [[OBJECTS]], i64 0, i64 1
  // CHECK: invoke void @_ZN1YC1Ev
  // CHECK: [[OBJECT1:%[a-zA-Z0-9.]+]] = invoke noundef ptr @_ZNK1YcvP11objc_objectEv{{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
  // CHECK: store ptr [[OBJECT1]], ptr [[ELEMENT1]]

  // Build the array
  // CHECK: {{invoke.*@objc_msgSend}}{{.*}} [ "clang.arc.attachedcall"(ptr @llvm.objc.retainAutoreleasedReturnValue) ]
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
  // CHECK-NEXT: call void @llvm.lifetime.end.p0(ptr [[ARR]])
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

