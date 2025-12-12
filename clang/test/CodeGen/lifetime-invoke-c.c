// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s -O1 -disable-llvm-passes | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s -O1 -disable-llvm-passes -fexceptions | FileCheck %s --check-prefix=EXCEPTIONS

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
  // CHECK: call void @gen(ptr{{.*}} sret(%struct.Trivial){{.*}} %[[AGG1]])
  // CHECK: call void @func(ptr{{.*}} %[[AGG1]])
  // CHECK: call void @llvm.lifetime.start.p0(ptr %[[AGG2]])
  // CHECK: call void @gen(ptr{{.*}} sret(%struct.Trivial){{.*}} %[[AGG2]])
  // CHECK: call void @func(ptr{{.*}} %[[AGG2]])
  // CHECK: call void @llvm.lifetime.end.p0(ptr %[[AGG2]])
  // CHECK: call void @llvm.lifetime.end.p0(ptr %[[AGG1]])
  // CHECK: call void @cleanup

  // EXCEPTIONS: %[[AGG1:.*]] = alloca %struct.Trivial
  // EXCEPTIONS: %[[AGG2:.*]] = alloca %struct.Trivial
  // EXCEPTIONS-NOT: call void @llvm.lifetime.start.p0(ptr %[[AGG1]])
  // EXCEPTIONS-NOT: call void @llvm.lifetime.start.p0(ptr %[[AGG2]])
  // EXCEPTIONS-NOT: call void @llvm.lifetime.end.p0(ptr %[[AGG2]])
  // EXCEPTIONS-NOT: call void @llvm.lifetime.end.p0(ptr %[[AGG1]])
  func(gen());
  func(gen());
}
