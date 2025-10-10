// Test issue 113843 and 118922 fix: do not elide hlfir.elemental final as_expr
// copy if this is not the last operation.
// RUN: fir-opt %s --bufferize-hlfir | FileCheck %s

func.func @_QMmPbug(%val: !fir.char<1>, %var: !fir.ref<!fir.array<10x!fir.char<1>>>) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %expr = hlfir.elemental %1 typeparams %c1 unordered : (!fir.shape<1>, index) -> !hlfir.expr<10x!fir.char<1>> {
  ^bb0(%arg2: index):
    %alloc = fir.allocmem !fir.char<1>
    fir.store %val to %alloc : !fir.heap<!fir.char<1>>
    %addr = fir.convert %alloc: (!fir.heap<!fir.char<1>>) -> !fir.ref<!fir.char<1>>
    %9 = hlfir.as_expr %addr : (!fir.ref<!fir.char<1>>) -> !hlfir.expr<!fir.char<1>>
    fir.freemem  %alloc : !fir.heap<!fir.char<1>>
    hlfir.yield_element %9 : !hlfir.expr<!fir.char<1>>
  }
  hlfir.assign %expr to %var : !hlfir.expr<10x!fir.char<1>>, !fir.ref<!fir.array<10x!fir.char<1>>>
  hlfir.destroy %expr : !hlfir.expr<10x!fir.char<1>>
  return
}

// CHECK-LABEL:   func.func @_QMmPbug(
// CHECK-SAME:                        %[[VAL_0:.*]]: !fir.char<1>,
// CHECK-SAME:                        %[[VAL_1:.*]]: !fir.ref<!fir.array<10x!fir.char<1>>>) {
// CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.char<1> {bindc_name = ".tmp"}
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_6:.*]] = fir.allocmem !fir.array<10x!fir.char<1>> {bindc_name = ".tmp.array", uniq_name = ""}
// CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]](%[[VAL_5]]) typeparams %[[VAL_3]] {uniq_name = ".tmp.array"} : (!fir.heap<!fir.array<10x!fir.char<1>>>, !fir.shape<1>, index) -> (!fir.heap<!fir.array<10x!fir.char<1>>>, !fir.heap<!fir.array<10x!fir.char<1>>>)
// CHECK:           %[[VAL_9:.*]] = arith.constant 1 : index
// CHECK:           fir.do_loop %[[VAL_10:.*]] = %[[VAL_9]] to %[[VAL_4]] step %[[VAL_9]] unordered {
// CHECK:             %[[VAL_11:.*]] = fir.allocmem !fir.char<1>
// CHECK:             fir.store %[[VAL_0]] to %[[VAL_11]] : !fir.heap<!fir.char<1>>
// CHECK:             %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (!fir.heap<!fir.char<1>>) -> !fir.ref<!fir.char<1>>
// CHECK:             %[[VAL_13:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_2]] typeparams %[[VAL_13]] {uniq_name = ".tmp"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
// CHECK:             hlfir.assign %[[VAL_12]] to %[[VAL_15]]#0 temporary_lhs : !fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>
// CHECK:             %[[VAL_14:.*]] = arith.constant false
// CHECK:             %[[VAL_16:.*]] = fir.undefined tuple<!fir.ref<!fir.char<1>>, i1>
// CHECK:             %[[VAL_17:.*]] = fir.insert_value %[[VAL_16]], %[[VAL_14]], [1 : index] : (tuple<!fir.ref<!fir.char<1>>, i1>, i1) -> tuple<!fir.ref<!fir.char<1>>, i1>
// CHECK:             %[[VAL_18:.*]] = fir.insert_value %[[VAL_17]], %[[VAL_15]]#0, [0 : index] : (tuple<!fir.ref<!fir.char<1>>, i1>, !fir.ref<!fir.char<1>>) -> tuple<!fir.ref<!fir.char<1>>, i1>
// CHECK:             fir.freemem %[[VAL_11]] : !fir.heap<!fir.char<1>>
// CHECK:             %[[VAL_19:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_10]])  typeparams %[[VAL_3]] : (!fir.heap<!fir.array<10x!fir.char<1>>>, index, index) -> !fir.ref<!fir.char<1>>
// CHECK:             hlfir.assign %[[VAL_15]]#0 to %[[VAL_19]] temporary_lhs : !fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>
// CHECK:           }
