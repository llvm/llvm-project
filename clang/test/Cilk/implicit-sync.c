// Verify that a sync is added implicitly at all exits to a function
// when -fcilkplus is specified.
//
// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -fcilkplus -ftapir=none -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

int foo(int n);

// CHECK-LABEL: doesnt_need_implicit_sync(
void doesnt_need_implicit_sync(int n) {
  foo(n);
  // CHECK-NOT: sync
  // CHECK: ret void
}

// CHECK-LABEL: needs_implicit_sync(
void needs_implicit_sync(int n) {
  // CHECK: %[[SYNCREGION:.+]] = call token @llvm.syncregion.start()
  // CHECK: detach within %[[SYNCREGION]]
  _Cilk_spawn foo(n);
  foo(n);
  // CHECK: sync within %[[SYNCREGION]], label %[[SYNCCONT:.+]]
  // CHECK: [[SYNCCONT]]:
  // CHECK-NEXT: ret void
}

// CHECK-LABEL: nested_implicit_sync(
void needs_nested_implicit_sync(int n) {
  // CHECK: %[[SYNCREGION:.+]] = call token @llvm.syncregion.start()
  // CHECK: detach within %[[SYNCREGION]]
  _Cilk_spawn {
    // CHECK-NOT: call token @llvm.syncregion.start()
    foo(n);
  }
  // CHECK: detach within %[[SYNCREGION]]
  _Cilk_spawn {
    // CHECK: %[[SYNCREGIONINNER:.+]] = call token @llvm.syncregion.start()
    // CHECK: detach within %[[SYNCREGIONINNER]]
    _Cilk_spawn foo(n);
    // CHECK: sync within %[[SYNCREGIONINNER]], label %[[INNERSYNCCONT:.+]]
    // CHECK: [[INNERSYNCCONT]]:
    // CHECK-NEXT: reattach within %[[SYNCREGION]]
  }
  foo(n);
  // CHECK: sync within %[[SYNCREGION]], label %[[SYNCCONT:.+]]
  // CHECK: [[SYNCCONT]]:
  // CHECK-NEXT: ret void
}
