// RUN: %clang_cc1 -O3 -triple x86_64-apple-macosx10.10 -main-file-name c-unreachable-after-switch.c %s -o - -emit-llvm -fprofile-instr-generate | FileCheck %s

// CHECK: @[[C:__profc_foo]] = private global [3 x i64] zeroinitializer

// CHECK-LABEL: @foo()
// CHECK: store {{.*}} @[[C]], i64 0, i64 0
void foo() {
  // CHECK: store {{.*}} @[[C]], i64 0, i64 2
  switch (0) {
  default:
    return;
  }
  // We shouldn't emit the unreachable counter. This used to crash in GlobalDCE.
  // CHECK-NOT: store {{.*}} @[[SWC]], i64 0, i64 1}
}
