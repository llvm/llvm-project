// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -fcilkplus -ftapir=none -S -emit-llvm -o - | FileCheck %s

int bar();
int baz(int);

// CHECK-LABEL: syncreg_spawn(
int syncreg_spawn(int n) {
  // CHECK: %[[SYNCREGTOP:.+]] = call token @llvm.syncregion.start()
  // CHECK: detach within %[[SYNCREGTOP]], label %[[DETACHEDBAR:.+]],
  // CHECK: [[DETACHEDBAR]]:
  // CHECK-NEXT: call i32 (...) @bar
  int x = _Cilk_spawn bar();
  // CHECK: detach within %[[SYNCREGTOP]], label %[[DETACHEDBAZ:.+]],
  // CHECK: [[DETACHEDBAZ]]:
  // CHECK-NEXT: call i32 @baz
  int y = _Cilk_spawn baz(n);
  // CHECK: sync within %[[SYNCREGTOP]], label %[[SYNCCONT:.+]]
  _Cilk_sync;
  // CHECK: [[SYNCCONT]]:
  // CHECK: add
  // CHECK: ret
  return x+y;
}

// CHECK-LABEL: syncreg_loop(
int syncreg_loop(int n) {
  // CHECK: %[[SYNCREGLOOP:.+]] = call token @llvm.syncregion.start()
  // CHECK-DAG: detach within %[[SYNCREGLOOP]]
  // CHECK-DAG: sync within %[[SYNCREGLOOP]]
  _Cilk_for(int i = 0; i < n; ++i) {
    baz(i);
  }
}  

// CHECK-LABEL: mixed_spawn_and_loop(
int mixed_spawn_and_loop(int n) {
  // CHECK: %[[SYNCREGTOP:.+]] = call token @llvm.syncregion.start()
  // CHECK: %[[SYNCREGLOOP:.+]] = call token @llvm.syncregion.start()
  // CHECK: detach within %[[SYNCREGTOP]]
  int x = _Cilk_spawn bar();
  // CHECK-DAG: detach within %[[SYNCREGLOOP]]
  // CHECK-DAG: sync within %[[SYNCREGLOOP]]
  _Cilk_for(int i = 0; i < n; ++i) {
    baz(i);
  }
  // CHECK: detach within %[[SYNCREGTOP]]
  int y = _Cilk_spawn baz(n);
  // CHECK: sync within %[[SYNCREGTOP]]
  _Cilk_sync;
  return x+y;
}
