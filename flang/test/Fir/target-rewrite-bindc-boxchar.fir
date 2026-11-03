// A BIND(C) procedure has no hidden length argument for its CHARACTER dummy
// arguments: the C side passes only the address, and the length of an
// interoperable CHARACTER dummy comes from its declaration (assumed length is
// passed by descriptor). Other procedures keep the trailing length.
// RUN: fir-opt --target-rewrite="target=x86_64-unknown-linux-gnu" %s | FileCheck %s
// RUN: fir-opt --target-rewrite="target=s390x-unknown-linux-gnu" %s | FileCheck %s

// CHECK-LABEL: func.func private @dlsym(!fir.ref<i64>, !fir.ref<!fir.char<1,?>>) -> !fir.ref<i64>
func.func private @dlsym(!fir.ref<i64>, !fir.boxchar<1>) -> !fir.ref<i64> attributes {fir.bindc_name = "dlsym", fir.proc_attrs = #fir.proc_attrs<bind_c>}
// CHECK-LABEL: func.func private @notbindc(!fir.ref<!fir.char<1,?>>, i64)
func.func private @notbindc(!fir.boxchar<1>)

// Direct calls, also without procedure attributes on the call (an earlier
// rewrite may have dropped them): the callee decides.
// CHECK-LABEL: func.func @calls(
// CHECK-SAME: %[[H:.*]]: !fir.ref<i64>, %[[A:.*]]: !fir.ref<!fir.char<1,?>>, %[[L:.*]]: i64)
// CHECK: %[[S:.*]] = fir.emboxchar %[[A]], %[[L]]
func.func @calls(%h : !fir.ref<i64>, %s : !fir.boxchar<1>) {
  // CHECK: %[[U1:.*]]:2 = fir.unboxchar %[[S]]
  // CHECK: fir.call @dlsym(%[[H]], %[[U1]]#0) proc_attrs<bind_c> : (!fir.ref<i64>, !fir.ref<!fir.char<1,?>>) -> !fir.ref<i64>
  %0 = fir.call @dlsym(%h, %s) proc_attrs<bind_c> : (!fir.ref<i64>, !fir.boxchar<1>) -> !fir.ref<i64>
  // CHECK: %[[U2:.*]]:2 = fir.unboxchar %[[S]]
  // CHECK: fir.call @dlsym(%[[H]], %[[U2]]#0) : (!fir.ref<i64>, !fir.ref<!fir.char<1,?>>) -> !fir.ref<i64>
  %1 = fir.call @dlsym(%h, %s) : (!fir.ref<i64>, !fir.boxchar<1>) -> !fir.ref<i64>
  // CHECK: %[[U3:.*]]:2 = fir.unboxchar %[[S]]
  // CHECK: fir.call @notbindc(%[[U3]]#0, %[[U3]]#1) : (!fir.ref<!fir.char<1,?>>, i64) -> ()
  fir.call @notbindc(%s) : (!fir.boxchar<1>) -> ()
  // A direct call follows the callee even if the call claims BIND(C).
  // CHECK: %[[U4:.*]]:2 = fir.unboxchar %[[S]]
  // CHECK: fir.call @notbindc(%[[U4]]#0, %[[U4]]#1) proc_attrs<bind_c> : (!fir.ref<!fir.char<1,?>>, i64) -> ()
  fir.call @notbindc(%s) proc_attrs<bind_c> : (!fir.boxchar<1>) -> ()
  return
}

// Indirect call: its procedure attributes decide.
// CHECK-LABEL: func.func @indirect(
// CHECK: fir.call %{{.*}}(%{{.*}}#0) proc_attrs<bind_c> : (!fir.ref<!fir.char<1,?>>) -> ()
// CHECK: fir.call %{{.*}}(%{{.*}}#0, %{{.*}}#1) : (!fir.ref<!fir.char<1,?>>, i64) -> ()
func.func @indirect(%p : () -> (), %s : !fir.boxchar<1>) {
  %f = fir.convert %p : (() -> ()) -> ((!fir.boxchar<1>) -> ())
  fir.call %f(%s) proc_attrs<bind_c> : (!fir.boxchar<1>) -> ()
  fir.call %f(%s) : (!fir.boxchar<1>) -> ()
  return
}

// Definition: the body gets a boxchar of the address and an undefined length.
// CHECK-LABEL: func.func @bindcdef(
// CHECK-SAME: %[[A:.*]]: !fir.ref<!fir.char<1,?>>, %[[N:.*]]: i32{{.*}}) -> {{.*}}i32
// CHECK: %[[L:.*]] = fir.undefined index
// CHECK: %[[B:.*]] = fir.emboxchar %[[A]], %[[L]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
// CHECK: fir.unboxchar %[[B]]
func.func @bindcdef(%s : !fir.boxchar<1>, %n : i32) -> i32 attributes {fir.bindc_name = "bindcdef", fir.proc_attrs = #fir.proc_attrs<bind_c>} {
  %0:2 = fir.unboxchar %s : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1>>, index)
  return %n : i32
}

// Address of a BIND(C) function: its type has no length either.
// CHECK-LABEL: func.func @addr(
// CHECK: fir.address_of(@bindcdef) : (!fir.ref<!fir.char<1,?>>, i32) -> i32
func.func @addr() -> ((!fir.boxchar<1>, i32) -> i32) {
  %0 = fir.address_of(@bindcdef) : (!fir.boxchar<1>, i32) -> i32
  return %0 : (!fir.boxchar<1>, i32) -> i32
}
