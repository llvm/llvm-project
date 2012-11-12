// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fapple-kext -fno-rtti -emit-llvm -o - %s | FileCheck %s

// CHECK: define void @_ZN2B1D0Ev
// CHECK: [[T1:%.*]] = load void (%struct.B1*)** getelementptr inbounds (void (%struct.B1*)** bitcast ([5 x i8*]* @_ZTV2B1 to void (%struct.B1*)**), i64 2)
// CHECK-NEXT: call void [[T1]](%struct.B1* [[T2:%.*]])
// CHECK: define void @_Z6DELETEP2B1
// CHECK: [[T3:%.*]] = load void (%struct.B1*)** getelementptr inbounds (void (%struct.B1*)** bitcast ([5 x i8*]* @_ZTV2B1 to void (%struct.B1*)**), i64 2)
// CHECK-NEXT:  call void [[T3]](%struct.B1* [[T4:%.*]])


struct B1 { 
  virtual ~B1(); 
};

B1::~B1() {}

void DELETE(B1 *pb1) {
  pb1->B1::~B1();
}
