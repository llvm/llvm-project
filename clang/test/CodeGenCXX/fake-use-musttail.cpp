// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -fextend-variable-liveness -o - %s | FileCheck %s

/// Tests that when we have fake uses in a function ending in a musttail call,
/// we emit the fake uses and their corresponding loads immediately prior to the
/// tail call.

extern "C" char *bar(int *);

// CHECK-LABEL: define dso_local ptr @foo(
// CHECK-SAME:    ptr noundef [[E:%.*]])
// CHECK-NEXT:  [[ENTRY:.*:]]
// CHECK-NEXT:    [[E_ADDR:%.*]] = alloca ptr, align 8
// CHECK-NEXT:    store ptr [[E]], ptr [[E_ADDR]], align 8
// CHECK-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[E_ADDR]], align 8
// CHECK-NEXT:    [[FAKE_USE:%.*]] = load ptr, ptr [[E_ADDR]]
// CHECK-NEXT:    notail call void (...) @llvm.fake.use(ptr [[FAKE_USE]])
// CHECK-NEXT:    [[CALL:%.*]] = musttail call ptr @bar(ptr noundef [[TMP0]])
// CHECK-NEXT:    ret ptr [[CALL]]

// CHECK:       [[BB1:.*:]]
// CHECK-NEXT:    [[FAKE_USE1:%.*]] = load ptr, ptr [[E_ADDR]]
// CHECK-NEXT:    notail call void (...) @llvm.fake.use(ptr [[FAKE_USE1]])
// CHECK-NEXT:    ret ptr undef
//
extern "C" const char *foo(int *e) {
  [[clang::musttail]] return bar(e);
}
