// Test CSE for hlfir.elemental.
// RUN: fir-opt %s --cse | FileCheck %s

// Check that the CSE does not optimize the hlfir.elemental
// without handling the associated hlfir.destroy's, otherwise,
// the same temp might be freed twice causing double-free error.
func.func @_QFPtest(%arg0: !fir.boxchar<1> {fir.bindc_name = "a"}) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %0:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %1 = fir.convert %0#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<10x!fir.char<1>>>
  %2 = fir.shape %c10 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %1(%2) typeparams %c1 {uniq_name = "_QFFtestEa"} : (!fir.ref<!fir.array<10x!fir.char<1>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<10x!fir.char<1>>>, !fir.ref<!fir.array<10x!fir.char<1>>>)
  %4 = fir.shape %c10 : (index) -> !fir.shape<1>
  %5 = hlfir.elemental %4 typeparams %c1 unordered : (!fir.shape<1>, index) -> !hlfir.expr<10x!fir.char<1>> {
  ^bb0(%arg1: index):
    %14 = fir.convert %arg1 : (index) -> i32
    %15 = fir.convert %14 : (i32) -> i64
    %16 = hlfir.designate %3#0 (%15)  typeparams %c1 : (!fir.ref<!fir.array<10x!fir.char<1>>>, i64, index) -> !fir.ref<!fir.char<1>>
    hlfir.yield_element %16 : !fir.ref<!fir.char<1>>
  }
  %6:3 = hlfir.associate %5(%4) typeparams %c1 {uniq_name = "adapt.valuebyref"} : (!hlfir.expr<10x!fir.char<1>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<10x!fir.char<1>>>, !fir.ref<!fir.array<10x!fir.char<1>>>, i1)
  hlfir.end_associate %6#1, %6#2 : !fir.ref<!fir.array<10x!fir.char<1>>>, i1
  hlfir.destroy %5 : !hlfir.expr<10x!fir.char<1>>
  %9 = fir.shape %c10 : (index) -> !fir.shape<1>
  %10 = hlfir.elemental %9 typeparams %c1 unordered : (!fir.shape<1>, index) -> !hlfir.expr<10x!fir.char<1>> {
  ^bb0(%arg1: index):
    %14 = fir.convert %arg1 : (index) -> i32
    %15 = fir.convert %14 : (i32) -> i64
    %16 = hlfir.designate %3#0 (%15)  typeparams %c1 : (!fir.ref<!fir.array<10x!fir.char<1>>>, i64, index) -> !fir.ref<!fir.char<1>>
    hlfir.yield_element %16 : !fir.ref<!fir.char<1>>
  }
  %11:3 = hlfir.associate %10(%9) typeparams %c1 {uniq_name = "adapt.valuebyref"} : (!hlfir.expr<10x!fir.char<1>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<10x!fir.char<1>>>, !fir.ref<!fir.array<10x!fir.char<1>>>, i1)
  hlfir.end_associate %11#1, %11#2 : !fir.ref<!fir.array<10x!fir.char<1>>>, i1
  hlfir.destroy %10 : !hlfir.expr<10x!fir.char<1>>
  return
}

// CHECK-LABEL:   func.func @_QFPtest(
// CHECK-SAME:                        %[[VAL_0:.*]]: !fir.boxchar<1> {fir.bindc_name = "a"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_3:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_4:.*]] = fir.convert %[[VAL_3]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<10x!fir.char<1>>>
// CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_4]](%[[VAL_5]]) typeparams %[[VAL_1]] {uniq_name = "_QFFtestEa"} : (!fir.ref<!fir.array<10x!fir.char<1>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<10x!fir.char<1>>>, !fir.ref<!fir.array<10x!fir.char<1>>>)
// CHECK:           %[[VAL_7:.*]] = hlfir.elemental %[[VAL_5]] typeparams %[[VAL_1]] unordered : (!fir.shape<1>, index) -> !hlfir.expr<10x!fir.char<1>> {
// CHECK:           ^bb0(%[[VAL_8:.*]]: index):
// CHECK:             %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (index) -> i32
// CHECK:             %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> i64
// CHECK:             %[[VAL_11:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_10]])  typeparams %[[VAL_1]] : (!fir.ref<!fir.array<10x!fir.char<1>>>, i64, index) -> !fir.ref<!fir.char<1>>
// CHECK:             hlfir.yield_element %[[VAL_11]] : !fir.ref<!fir.char<1>>
// CHECK:           }
// CHECK:           %[[VAL_12:.*]]:3 = hlfir.associate %[[VAL_7]](%[[VAL_5]]) typeparams %[[VAL_1]] {uniq_name = "adapt.valuebyref"} : (!hlfir.expr<10x!fir.char<1>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<10x!fir.char<1>>>, !fir.ref<!fir.array<10x!fir.char<1>>>, i1)
// CHECK:           hlfir.end_associate %[[VAL_12]]#1, %[[VAL_12]]#2 : !fir.ref<!fir.array<10x!fir.char<1>>>, i1
// CHECK:           hlfir.destroy %[[VAL_7]] : !hlfir.expr<10x!fir.char<1>>
// CHECK:           %[[VAL_13:.*]] = hlfir.elemental %[[VAL_5]] typeparams %[[VAL_1]] unordered : (!fir.shape<1>, index) -> !hlfir.expr<10x!fir.char<1>> {
// CHECK:           ^bb0(%[[VAL_14:.*]]: index):
// CHECK:             %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (index) -> i32
// CHECK:             %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i32) -> i64
// CHECK:             %[[VAL_17:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_16]])  typeparams %[[VAL_1]] : (!fir.ref<!fir.array<10x!fir.char<1>>>, i64, index) -> !fir.ref<!fir.char<1>>
// CHECK:             hlfir.yield_element %[[VAL_17]] : !fir.ref<!fir.char<1>>
// CHECK:           }
// CHECK:           %[[VAL_18:.*]]:3 = hlfir.associate %[[VAL_13]](%[[VAL_5]]) typeparams %[[VAL_1]] {uniq_name = "adapt.valuebyref"} : (!hlfir.expr<10x!fir.char<1>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<10x!fir.char<1>>>, !fir.ref<!fir.array<10x!fir.char<1>>>, i1)
// CHECK:           hlfir.end_associate %[[VAL_18]]#1, %[[VAL_18]]#2 : !fir.ref<!fir.array<10x!fir.char<1>>>, i1
// CHECK:           hlfir.destroy %[[VAL_13]] : !hlfir.expr<10x!fir.char<1>>
// CHECK:           return
// CHECK:         }
