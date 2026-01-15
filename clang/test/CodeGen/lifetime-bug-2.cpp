// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -emit-llvm -disable-llvm-passes -target-cpu x86-64 -fexceptions -fcxx-exceptions -x c++ %s -o - 2>&1 | FileCheck %s

extern "C" {
struct X {};

X f1(X x);
X f2(X x);

void foo(){
  try{
  f2(f1(X{}));
  } catch(int e){
    return;
  }
  return;
}
}
// CHECK-LABEL: define{{.*}} void @foo
// CHECK: [[TMP1:%.*]] = alloca %struct.X
// CHECK: [[TMP2:%.*]] = alloca %struct.X
// CHECK: llvm.lifetime.start.p0(ptr [[TMP1]])
// CHECK-NEXT: llvm.lifetime.start.p0(ptr [[TMP2]])
// CHECK-NEXT: invoke void @f1
// CHECK-NEXT: to label %[[CONT:.*]] unwind label %[[LPAD1:.*]]
//
// CHECK: [[CONT]]:
// CHECK-NEXT: @llvm.lifetime.end.p0(ptr [[TMP2]])
// CHECK-NEXT: invoke void @f2
// CHECK-NEXT: to label %[[CONT2:.*]] unwind label %[[LPAD2:.*]]
//
// CHECK: [[CONT2]]:
// CHECK-NEXT: lifetime.end.p0(ptr [[TMP1]])
//
// CHECK: [[LPAD1]]:
// CHECK-NEXT: landingpad
// CHECK: llvm.lifetime.end.p0(ptr [[TMP1]])