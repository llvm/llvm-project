// RUN: fir-opt -flang-licm --split-input-file %s | FileCheck %s

// Tests checking that Flang's LICM hoists the load of a saved local variable
// out of a loop with a call only if the call cannot invoke the procedure that
// owns the variable. All instances of the procedure share the variable.

// The RECURSIVE work may be invoked by bump, and modify x: the load is not
// hoisted.
// CHECK-LABEL: func.func @_QPwork_recursive(
// CHECK:         fir.do_loop
// CHECK-NEXT:      fir.call @_QPbump() : () -> ()
// CHECK-NEXT:      fir.load
fir.global internal @_QFwork_recursiveEx : i32 {
  %0 = fir.zero_bits i32
  fir.has_value %0 : i32
}
func.func @_QPwork_recursive(%n: index, %r: !fir.ref<i32>) attributes {fir.proc_attrs = #fir.proc_attrs<recursive>} {
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = fir.address_of(@_QFwork_recursiveEx) : !fir.ref<i32>
  %1 = fir.declare %0 uniq_name("_QFwork_recursiveEx") : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.do_loop %i = %c1 to %n step %c1 iter_args(%acc = %c0_i32) -> (i32) {
    fir.call @_QPbump() : () -> ()
    %3 = fir.load %1 : !fir.ref<i32>
    %4 = arith.addi %acc, %3 : i32
    fir.result %4 : i32
  }
  fir.store %2 to %r : !fir.ref<i32>
  return
}
func.func private @_QPbump()

// -----

// The NON_RECURSIVE work cannot be invoked by bump: the load is hoisted.
// CHECK-LABEL: func.func @_QPwork_non_recursive(
// CHECK:         %[[X:.*]] = fir.declare
// CHECK:         fir.load %[[X]] : !fir.ref<i32>
// CHECK:         fir.do_loop
// CHECK-NOT:       fir.load
// CHECK:           fir.call @_QPbump() : () -> ()
// CHECK-NOT:       fir.load
// CHECK:           fir.result
fir.global internal @_QFwork_non_recursiveEx : i32 {
  %0 = fir.zero_bits i32
  fir.has_value %0 : i32
}
func.func @_QPwork_non_recursive(%n: index, %r: !fir.ref<i32>) attributes {fir.proc_attrs = #fir.proc_attrs<non_recursive>} {
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = fir.address_of(@_QFwork_non_recursiveEx) : !fir.ref<i32>
  %1 = fir.declare %0 uniq_name("_QFwork_non_recursiveEx") : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.do_loop %i = %c1 to %n step %c1 iter_args(%acc = %c0_i32) -> (i32) {
    fir.call @_QPbump() : () -> ()
    %3 = fir.load %1 : !fir.ref<i32>
    %4 = arith.addi %acc, %3 : i32
    fir.result %4 : i32
  }
  fir.store %2 to %r : !fir.ref<i32>
  return
}
func.func private @_QPbump()

// -----

// The PURE pf cannot invoke the impure work: the load is hoisted.
// CHECK-LABEL: func.func @_QPwork_pure_callee(
// CHECK:         %[[X:.*]] = fir.declare
// CHECK:         fir.load %[[X]] : !fir.ref<i32>
// CHECK:         fir.do_loop
// CHECK-NOT:       fir.load
// CHECK:           fir.call @_QPpf() proc_attrs<pure> : () -> ()
// CHECK-NOT:       fir.load
// CHECK:           fir.result
fir.global internal @_QFwork_pure_calleeEx : i32 {
  %0 = fir.zero_bits i32
  fir.has_value %0 : i32
}
func.func @_QPwork_pure_callee(%n: index, %r: !fir.ref<i32>) {
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = fir.address_of(@_QFwork_pure_calleeEx) : !fir.ref<i32>
  %1 = fir.declare %0 uniq_name("_QFwork_pure_calleeEx") : (!fir.ref<i32>) -> !fir.ref<i32>
  %2 = fir.do_loop %i = %c1 to %n step %c1 iter_args(%acc = %c0_i32) -> (i32) {
    fir.call @_QPpf() proc_attrs<pure> : () -> ()
    %3 = fir.load %1 : !fir.ref<i32>
    %4 = arith.addi %acc, %3 : i32
    fir.result %4 : i32
  }
  fir.store %2 to %r : !fir.ref<i32>
  return
}
func.func private @_QPpf() attributes {fir.proc_attrs = #fir.proc_attrs<pure>}
