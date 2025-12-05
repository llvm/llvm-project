// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s -O1 -disable-llvm-passes -fexceptions | FileCheck %s

struct Trivial {
  int x[100];
};

void cleanup(int *p) {}
void func(struct Trivial t);
struct Trivial gen(void);

// CHECK-LABEL: define dso_local void @test()
void test() {
  int x __attribute__((cleanup(cleanup)));

  // CHECK: %[[AGG1:.*]] = alloca %struct.Trivial
  // CHECK: %[[AGG2:.*]] = alloca %struct.Trivial

  // CHECK: call void @llvm.lifetime.start.p0(ptr %[[AGG1]])
  // CHECK: invoke void @gen(ptr{{.*}} sret(%struct.Trivial){{.*}} %[[AGG1]])

  // CHECK: invoke void @func(ptr{{.*}} %[[AGG1]])
  // CHECK-NEXT: to label %[[CONT1:.*]] unwind label %[[LPAD1:.*]]

  // CHECK: [[CONT1]]:
  // CHECK-NOT: call void @llvm.lifetime.end.p0(ptr %[[AGG1]])

  // CHECK: call void @llvm.lifetime.start.p0(ptr %[[AGG2]])
  // CHECK: invoke void @gen(ptr{{.*}} sret(%struct.Trivial){{.*}} %[[AGG2]])
  // CHECK: invoke void @func(ptr{{.*}} %[[AGG2]])
  // CHECK-NEXT: to label %[[CONT2:.*]] unwind label %[[LPAD2:.*]]

  // CHECK: [[CONT2]]:
  // CHECK-DAG: call void @llvm.lifetime.end.p0(ptr %[[AGG2]])
  // CHECK-DAG: call void @llvm.lifetime.end.p0(ptr %[[AGG1]])

  // CHECK: [[LPAD1]]:
  // CHECK: landingpad
  // CHECK: br label %[[EHCLEANUP:.*]]

  // CHECK: [[LPAD2]]:
  // CHECK: landingpad
  // CHECK: call void @llvm.lifetime.end.p0(ptr %[[AGG2]])
  // CHECK: br label %[[EHCLEANUP]]

  // CHECK: [[EHCLEANUP]]:
  // CHECK: call void @llvm.lifetime.end.p0(ptr %[[AGG1]])
  // CHECK: call void @cleanup
  func(gen());
  func(gen());
}
