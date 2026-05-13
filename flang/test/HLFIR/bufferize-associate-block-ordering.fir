// RUN: fir-opt --bufferize-hlfir %s | FileCheck %s

func.func @shape_of_non_dominating_block(%cond: i1, %n: index) {
  %shape = fir.shape %n : (index) -> !fir.shape<1>
  %expr = hlfir.elemental %shape : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
  ^bb0(%i: index):
    %c0 = arith.constant 0 : i32
    hlfir.yield_element %c0 : i32
  }
  %assoc:3 = hlfir.associate %expr(%shape) {uniq_name = "test"} : (!hlfir.expr<?xi32>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>, i1)
  cf.cond_br %cond, ^bb1, ^bb2
^bb1:
  %s = hlfir.shape_of %expr : (!hlfir.expr<?xi32>) -> !fir.shape<1>
  cf.br ^bb3
^bb2:
  cf.br ^bb3
^bb3:
  hlfir.end_associate %assoc#1, %assoc#2 : !fir.ref<!fir.array<?xi32>>, i1
  hlfir.destroy %expr : !hlfir.expr<?xi32>
  return
}

// The shape_of in ^bb1 does not dominate the end_associate in ^bb3
// (^bb3 is reachable from ^bb2 as well), so allOtherUsesAreSafeForAssociate
// returns false and the associate takes the copy path.
// CHECK-LABEL: func.func @shape_of_non_dominating_block(
// CHECK-SAME:    %[[COND:.*]]: i1, %[[N:.*]]: index)
// CHECK:       %[[SHAPE:.*]] = fir.shape %[[N]]
// CHECK:       %[[ELEM_BUF:.*]] = fir.allocmem !fir.array<?xi32>, %[[N]] {{.*}}".tmp.array"
// CHECK:       fir.do_loop
// CHECK:       %[[COPY_BUF:.*]] = fir.allocmem !fir.array<?xi32>, %[[N]] {{.*}}".tmp"
// CHECK:       hlfir.assign %{{.*}} to %{{.*}} temporary_lhs
// CHECK:       cf.cond_br %[[COND]], ^bb1, ^bb2
// CHECK:     ^bb1:
// CHECK:       cf.br ^bb3
// CHECK:     ^bb2:
// CHECK:       cf.br ^bb3
// CHECK:     ^bb3:
// CHECK:       fir.freemem
// CHECK:       fir.freemem
// CHECK:       return
