/// Test the AArch64 ABI rewrite of a BIND(C), VALUE derived type argument that
/// is larger than 16 bytes. Such an argument is passed as a pointer to a copy
/// made by the caller, and is not marked byval.

// RUN: fir-opt --target-rewrite="target=aarch64-unknown-linux-gnu" %s | FileCheck %s

!big = !fir.type<big{i:!fir.array<128x!fir.char<1>>}>

/// The callee receives a pointer and loads the value from it.
// CHECK-LABEL: func.func @callee(
// CHECK-SAME:      %[[ARG:.*]]: !fir.ref<!fir.type<big{i:!fir.array<128x!fir.char<1>>}>> {llvm.align = 1 : i32}) {
// CHECK-NOT:     llvm.byval
// CHECK:         %[[REF:.*]] = fir.convert %[[ARG]]
// CHECK:         fir.load %[[REF]]
func.func @callee(%arg0: !big) {
  return
}

/// The caller stores the value into a temporary and passes its address.
// CHECK-LABEL: func.func @caller(
// CHECK-NOT:     llvm.byval
// CHECK:         %[[VAL:.*]] = fir.load
// CHECK:         %[[TMP:.*]] = fir.alloca !fir.type<big{i:!fir.array<128x!fir.char<1>>}>
// CHECK:         fir.store %[[VAL]] to %[[TMP]]
// CHECK:         fir.call @callee(%[[TMP]]) : (!fir.ref<!fir.type<big{i:!fir.array<128x!fir.char<1>>}>> {llvm.align = 1 : i32}) -> ()
func.func @caller(%arg0: !fir.ref<!big>) {
  %0 = fir.load %arg0 : !fir.ref<!big>
  fir.call @callee(%0) : (!big) -> ()
  return
}
