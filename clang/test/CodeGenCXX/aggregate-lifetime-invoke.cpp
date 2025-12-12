// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s -O1 -fexceptions -fcxx-exceptions | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s -O1 -fexceptions -fcxx-exceptions -sloppy-temporary-lifetimes | FileCheck %s

// COM: Note that this test case would break if we allowed tighter lifetimes to
// run when exceptions were enabled. If we make them work together this test
// will need to be updated.

extern "C" {

struct Trivial {
  int x[100];
};

void func_that_throws(Trivial t);

// CHECK-LABEL: define{{.*}} void @test()
void test() {
  // CHECK: %[[AGG1:.*]] = alloca %struct.Trivial
  // CHECK: %[[AGG2:.*]] = alloca %struct.Trivial

  // CHECK: invoke void @func_that_throws(ptr{{.*}} %[[AGG1]])
  // CHECK-NEXT: to label %[[CONT:.*]] unwind label %[[LPAD:.*]]

  // CHECK: [[CONT]]:
  // CHECK: invoke void @func_that_throws(ptr{{.*}} %[[AGG2]])
  // CHECK-NEXT: to label %[[CONT:.*]] unwind label %[[LPAD:.*]]

  // CHECK: [[LPAD]]:
  // CHECK: landingpad

  // CHECK-NOT: llvm.lifetime.start
  // CHECK-NOT: llvm.lifetime.end

  try {
    func_that_throws(Trivial{0});
    func_that_throws(Trivial{0});
  } catch (...) {
  }
}
} // end extern "C"
