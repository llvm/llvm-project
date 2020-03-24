// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -fcilkplus -fexceptions -ftapir=none -S -emit-llvm -o - | FileCheck %s
// expected-no-diagnostics

template<typename intT>
intT fib(intT n) {
  if (n < 2) return n;
  intT x = _Cilk_spawn fib(n - 1);
  intT y = fib(n - 2);
  _Cilk_sync;
  return (x + y);
}

long foo() {
  return fib(38);
}

// CHECK-LABEL: define {{.+}} @_Z3fibIiET_S0_(i32 %n)

// CHECK: detach within %[[SYNCREG:.+]], label %[[DETBLK:.+]], label %[[CONTBLK:.+]] unwind

// CHECK: [[DETBLK]]:
// CHECK: %[[RETVAL:.+]] = invoke i32 @_Z3fibIiET_S0_
// CHECK-NEXT: to label %[[INVOKECONT:.+]] unwind label %[[TASKUNWIND:.+]]

// CHECK: [[INVOKECONT]]:
// CHECK-NEXT: store i32 %[[RETVAL]], i32*
// CHECK-NEXT: reattach within %[[SYNCREG]], label %[[CONTBLK]]

// CHECK: [[CONTBLK]]:
// CHECK: %[[RETVAL2:.+]] = invoke i32 @_Z3fibIiET_S0_
// CHECK-NEXT: to label %[[INVOKECONT2:.+]] unwind label %[[TASKUNWIND2:.+]]

// CHECK: [[INVOKECONT2]]:
// CHECK-NEXT: store i32 %[[RETVAL2]]
// CHECK-NEXT: sync within %[[SYNCREG]]
