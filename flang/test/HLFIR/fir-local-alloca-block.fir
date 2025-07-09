// Tests that `fir.local` ops are able to provide an alloca block when required.

// RUN: fir-opt %s -convert-hlfir-to-fir | FileCheck %s

fir.local {type = local_init} @localizer : !fir.box<!fir.array<1xi32>> copy {
^bb0(%arg0: !fir.ref<!fir.box<!fir.array<1xi32>>>, %arg1: !fir.ref<!fir.box<!fir.array<1xi32>>>):
  %0 = fir.load %arg0 : !fir.ref<!fir.box<!fir.array<1xi32>>>
  hlfir.assign %0 to %arg1 : !fir.box<!fir.array<1xi32>>, !fir.ref<!fir.box<!fir.array<1xi32>>>
  fir.yield(%arg1 : !fir.ref<!fir.box<!fir.array<1xi32>>>)
}

func.func @foo() {
  %c1 = arith.constant 1 : index
  %0 = fir.alloca !fir.box<!fir.array<1xi32>>
  fir.do_concurrent {
    fir.do_concurrent.loop (%arg0) = (%c1) to (%c1) step (%c1) local(@localizer %0 -> %arg1 : !fir.ref<!fir.box<!fir.array<1xi32>>>) {
    }
  }
  return
}

// CHECK:  fir.local {type = local_init} @localizer : ![[TYPE:fir.box<!fir.array<1xi32>>]] copy {
// CHECK:    ^bb0(%[[VAL_0:.*]]: !fir.ref<![[TYPE]]>, %[[VAL_1:.*]]: !fir.ref<![[TYPE]]>):
// CHECK:      %[[VAL_2:.*]] = fir.alloca ![[TYPE]]
// CHECK:      %[[VAL_3:.*]] = fir.load %[[VAL_0]] : !fir.ref<![[TYPE]]>
// CHECK:      %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:      %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_3]], %[[VAL_4]] : (![[TYPE]], index) -> (index, index, index)
// CHECK:      %[[VAL_6:.*]] = fir.load %[[VAL_1]] : !fir.ref<![[TYPE]]>
// CHECK:      fir.store %[[VAL_6]] to %[[VAL_2]] : !fir.ref<![[TYPE]]>
// CHECK:      %[[VAL_10:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<![[TYPE]]>) -> !fir.ref<!fir.box<none>>
// CHECK:      %[[VAL_11:.*]] = fir.convert %[[VAL_3]] : (![[TYPE]]) -> !fir.box<none>
// CHECK:      fir.call @_FortranAAssign(%[[VAL_10]], %[[VAL_11]], %{{.*}}, %{{.*}})
// CHECK:      fir.yield(%[[VAL_1]] : !fir.ref<![[TYPE]]>)
// CHECK:  }
