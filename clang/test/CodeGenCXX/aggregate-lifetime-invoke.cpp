// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s -O1 -fexceptions -fcxx-exceptions | FileCheck %s

struct Trivial {
  int x[100];
};

void func_that_throws(Trivial t);

// CHECK-LABEL: define dso_local void @_Z4testv(){{.*}} personality ptr @__gxx_personality_v0
void test() {
  // CHECK: %[[AGG1:.*]] = alloca %struct.Trivial, align
  // CHECK: %[[AGG2:.*]] = alloca %struct.Trivial, align

  // CHECK: call void @llvm.lifetime.start.p0(ptr nonnull %[[AGG1]])
  // CHECK: invoke void @_Z16func_that_throws7Trivial(ptr noundef nonnull byval(%struct.Trivial) align 8 %[[AGG1]])
  // CHECK-NEXT: to label %[[CONT1:.*]] unwind label %[[LPAD1:.*]]
  
  // CHECK: [[CONT1]]:
  // CHECK-NEXT: call void @llvm.lifetime.start.p0(ptr nonnull %[[AGG2]])
  // CHECK: invoke void @_Z16func_that_throws7Trivial(ptr noundef nonnull byval(%struct.Trivial) align 8 %[[AGG2]])
  // CHECK-NEXT: to label %[[CONT2:.*]] unwind label %[[LPAD2:.*]]

  // CHECK: [[CONT2]]:
  // CHECK-DAG: call void @llvm.lifetime.end.p0(ptr nonnull %[[AGG2]])
  // CHECK-DAG: call void @llvm.lifetime.end.p0(ptr nonnull %[[AGG1]])
  // CHECK: br label %[[TRY_CONT:.*]]

  // CHECK: [[LPAD1]]:
  // CHECK: landingpad
  // CHECK: br label %[[EHCLEANUP:.*]]

  // CHECK: [[LPAD2]]:
  // CHECK: landingpad
  // CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[AGG2]])
  // CHECK: br label %[[EHCLEANUP]]

  // CHECK: [[EHCLEANUP]]:
  // CHECK: call void @llvm.lifetime.end.p0(ptr nonnull %[[AGG1]])
  // CHECK: call ptr @__cxa_begin_catch
  try {
    func_that_throws(Trivial{0});
    func_that_throws(Trivial{0});
  } catch (...) {
  }
}
