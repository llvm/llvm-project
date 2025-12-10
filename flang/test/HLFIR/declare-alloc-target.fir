// // RUN: fir-opt %s -convert-hlfir-to-fir | FileCheck %s
// Check that the fir.box allocatable attribute is dropped when assigning to
// the target of an allocatable (and not to the allocatable itself directly).
// Otherwise, the runtime may trigger reallocation instead of trimming/padding.

func.func @repro(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>, %arg1: !fir.ref<!fir.array<1x!fir.char<1,7>>>) {
  %0 = fir.load %arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
  %1:2 = hlfir.declare %0 {uniq_name = "_QFreproEx"} : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.box<!fir.array<?x!fir.char<1,?>>>)
  hlfir.assign %arg1 to %1#0 : !fir.ref<!fir.array<1x!fir.char<1,7>>>, !fir.box<!fir.array<?x!fir.char<1,?>>>
  return
}
// CHECK-LABEL:   func.func @repro(
// CHECK-SAME:                     %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>,
// CHECK-SAME:                     %[[ARG1:.*]]: !fir.ref<!fir.array<1x!fir.char<1,7>>>) {
// CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.array<?x!fir.char<1,?>>>
// CHECK:           %[[VAL_1:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
// CHECK:           %[[VAL_2:.*]] = fir.declare %[[VAL_1]] {uniq_name = "_QFreproEx"} : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>) -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>
// CHECK:           %[[VAL_3:.*]] = fir.rebox %[[VAL_2]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
// CHECK:           fir.store %[[VAL_3]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.array<?x!fir.char<1,?>>>>
// CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_0]] : (!fir.ref<!fir.box<!fir.array<?x!fir.char<1,?>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           fir.call @_FortranAAssign(%[[VAL_11]],
