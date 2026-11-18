// RUN: fir-opt %s -split-input-file -o /dev/null --mlir-disable-threading \
// RUN:   -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis-modref))' \
// RUN:   2>&1 | FileCheck %s

// Test fir.call modref for an allocation whose address is stored into the
// descriptor of a saved local allocatable. The allocation becomes the data of
// the saved local, so a call that may invoke the procedure that owns the
// saved local may access it.

// subroutine work(n)
//   integer :: n
//   integer, allocatable, save :: a
//   a = n         ! a is not allocated: the assignment allocates a new buffer
//   call bump()   ! may call work, which may access a
//   call pf()     ! pure
// end subroutine

// CHECK-LABEL: Testing : "_QPwork"
// CHECK: bump_call -> new#0: ModRef{{$}}
// CHECK: pf_call -> new#0: NoModRef{{$}}
fir.global internal @_QFworkEa : !fir.box<!fir.heap<i32>> {
  %0 = fir.zero_bits !fir.heap<i32>
  %1 = fir.embox %0 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  fir.has_value %1 : !fir.box<!fir.heap<i32>>
}
func.func @_QPwork(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@_QFworkEa) : !fir.ref<!fir.box<!fir.heap<i32>>>
  %2 = fir.declare %1 uniq_name("_QFworkEa") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  %3 = fir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QFworkEn") : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %4 = fir.load %3 : !fir.ref<i32>
  %5 = fir.allocmem i32 {test.ptr = "new"}
  fir.store %4 to %5 : !fir.heap<i32>
  %6 = fir.embox %5 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  fir.store %6 to %2 : !fir.ref<!fir.box<!fir.heap<i32>>>
  fir.call @_QPbump() {test.ptr = "bump_call"} : () -> ()
  fir.call @_QPpf() proc_attrs<pure> {test.ptr = "pf_call"} : () -> ()
  return
}
func.func private @_QPbump()
func.func private @_QPpf() attributes {fir.proc_attrs = #fir.proc_attrs<pure>}

// -----

// The same with a NON_RECURSIVE work: bump cannot call work.

// CHECK-LABEL: Testing : "_QPwork"
// CHECK: bump_call -> new#0: NoModRef{{$}}
// CHECK: pf_call -> new#0: NoModRef{{$}}
fir.global internal @_QFworkEa : !fir.box<!fir.heap<i32>> {
  %0 = fir.zero_bits !fir.heap<i32>
  %1 = fir.embox %0 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  fir.has_value %1 : !fir.box<!fir.heap<i32>>
}
func.func @_QPwork(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}) attributes {fir.proc_attrs = #fir.proc_attrs<non_recursive>} {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@_QFworkEa) : !fir.ref<!fir.box<!fir.heap<i32>>>
  %2 = fir.declare %1 uniq_name("_QFworkEa") fortran_attrs<allocatable> : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  %3 = fir.declare %arg0 dummy_scope %0 arg 1 uniq_name("_QFworkEn") : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %4 = fir.load %3 : !fir.ref<i32>
  %5 = fir.allocmem i32 {test.ptr = "new"}
  fir.store %4 to %5 : !fir.heap<i32>
  %6 = fir.embox %5 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  fir.store %6 to %2 : !fir.ref<!fir.box<!fir.heap<i32>>>
  fir.call @_QPbump() {test.ptr = "bump_call"} : () -> ()
  fir.call @_QPpf() proc_attrs<pure> {test.ptr = "pf_call"} : () -> ()
  return
}
func.func private @_QPbump()
func.func private @_QPpf() attributes {fir.proc_attrs = #fir.proc_attrs<pure>}
