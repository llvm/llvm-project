// Test hlfir.eval_in_mem operation parse, verify (no errors), and unparse.

// RUN: fir-opt %s | fir-opt | FileCheck %s

func.func @_QPtest() {
  %c10 = arith.constant 10 : index
  %0 = fir.address_of(@_QFtestEx) : !fir.ref<!fir.array<10xf32>>
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2 = hlfir.eval_in_mem shape %1 : (!fir.shape<1>) -> !hlfir.expr<10xf32> {
  ^bb0(%arg0: !fir.ref<!fir.array<10xf32>>):
    %3 = fir.call @_QParray_func() fastmath<contract> : () -> !fir.array<10xf32>
    fir.save_result %3 to %arg0(%1) : !fir.array<10xf32>, !fir.ref<!fir.array<10xf32>>, !fir.shape<1>
  }
  hlfir.assign %2 to %0 : !hlfir.expr<10xf32>, !fir.ref<!fir.array<10xf32>>
  hlfir.destroy %2 : !hlfir.expr<10xf32>
  return
}
fir.global internal @_QFtestEx : !fir.array<10xf32>
func.func private @_QParray_func() -> !fir.array<10xf32>


func.func @_QPtest_char() {
  %c10 = arith.constant 10 : index
  %c5 = arith.constant 5 : index
  %0 = fir.address_of(@_QFtest_charEx) : !fir.ref<!fir.array<10x!fir.char<1,5>>>
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2 = hlfir.eval_in_mem shape %1 typeparams %c5 : (!fir.shape<1>, index) -> !hlfir.expr<10x!fir.char<1,5>> {
  ^bb0(%arg0: !fir.ref<!fir.array<10x!fir.char<1,5>>>):
    %3 = fir.call @_QPchar_array_func() fastmath<contract> : () -> !fir.array<10x!fir.char<1,5>>
    fir.save_result %3 to %arg0(%1) typeparams %c5 : !fir.array<10x!fir.char<1,5>>, !fir.ref<!fir.array<10x!fir.char<1,5>>>, !fir.shape<1>, index
  }
  hlfir.assign %2 to %0 : !hlfir.expr<10x!fir.char<1,5>>, !fir.ref<!fir.array<10x!fir.char<1,5>>>
  hlfir.destroy %2 : !hlfir.expr<10x!fir.char<1,5>>
  return
}

fir.global internal @_QFtest_charEx : !fir.array<10x!fir.char<1,5>>
func.func private @_QPchar_array_func() -> !fir.array<10x!fir.char<1,5>>

func.func @test_dynamic(%arg0: !fir.box<!fir.array<?xf32>>, %arg1: index) {
   %0 = fir.shape %arg1 : (index) -> !fir.shape<1>
   %1 = hlfir.eval_in_mem shape %0 : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
   ^bb0(%arg2: !fir.ref<!fir.array<?xf32>>):
     %2 = fir.call @_QPdyn_array_func(%arg1) : (index) -> !fir.array<?xf32>
     fir.save_result %2 to %arg2(%0) : !fir.array<?xf32>, !fir.ref<!fir.array<?xf32>>, !fir.shape<1>
   }
   hlfir.assign %1 to %arg0 : !hlfir.expr<?xf32>, !fir.box<!fir.array<?xf32>>
   hlfir.destroy %1 : !hlfir.expr<?xf32>
   return
}
func.func private @_QPdyn_array_func(index) -> !fir.array<?xf32>

// CHECK-LABEL:   func.func @_QPtest() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_1:.*]] = fir.address_of(@_QFtestEx) : !fir.ref<!fir.array<10xf32>>
// CHECK:           %[[VAL_2:.*]] = fir.shape %[[VAL_0]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_3:.*]] = hlfir.eval_in_mem shape %[[VAL_2]] : (!fir.shape<1>) -> !hlfir.expr<10xf32> {
// CHECK:           ^bb0(%[[VAL_4:.*]]: !fir.ref<!fir.array<10xf32>>):
// CHECK:             %[[VAL_5:.*]] = fir.call @_QParray_func() fastmath<contract> : () -> !fir.array<10xf32>
// CHECK:             fir.save_result %[[VAL_5]] to %[[VAL_4]](%[[VAL_2]]) : !fir.array<10xf32>, !fir.ref<!fir.array<10xf32>>, !fir.shape<1>
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_3]] to %[[VAL_1]] : !hlfir.expr<10xf32>, !fir.ref<!fir.array<10xf32>>
// CHECK:           hlfir.destroy %[[VAL_3]] : !hlfir.expr<10xf32>
// CHECK:           return
// CHECK:         }
// CHECK:         fir.global internal @_QFtestEx : !fir.array<10xf32>
// CHECK:         func.func private @_QParray_func() -> !fir.array<10xf32>

// CHECK-LABEL:   func.func @_QPtest_char() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 5 : index
// CHECK:           %[[VAL_2:.*]] = fir.address_of(@_QFtest_charEx) : !fir.ref<!fir.array<10x!fir.char<1,5>>>
// CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_0]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_4:.*]] = hlfir.eval_in_mem shape %[[VAL_3]] typeparams %[[VAL_1]] : (!fir.shape<1>, index) -> !hlfir.expr<10x!fir.char<1,5>> {
// CHECK:           ^bb0(%[[VAL_5:.*]]: !fir.ref<!fir.array<10x!fir.char<1,5>>>):
// CHECK:             %[[VAL_6:.*]] = fir.call @_QPchar_array_func() fastmath<contract> : () -> !fir.array<10x!fir.char<1,5>>
// CHECK:             fir.save_result %[[VAL_6]] to %[[VAL_5]](%[[VAL_3]]) typeparams %[[VAL_1]] : !fir.array<10x!fir.char<1,5>>, !fir.ref<!fir.array<10x!fir.char<1,5>>>, !fir.shape<1>, index
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_4]] to %[[VAL_2]] : !hlfir.expr<10x!fir.char<1,5>>, !fir.ref<!fir.array<10x!fir.char<1,5>>>
// CHECK:           hlfir.destroy %[[VAL_4]] : !hlfir.expr<10x!fir.char<1,5>>
// CHECK:           return
// CHECK:         }
// CHECK:         fir.global internal @_QFtest_charEx : !fir.array<10x!fir.char<1,5>>
// CHECK:         func.func private @_QPchar_array_func() -> !fir.array<10x!fir.char<1,5>>

// CHECK-LABEL:   func.func @test_dynamic(
// CHECK-SAME:                            %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>>,
// CHECK-SAME:                            %[[VAL_1:.*]]: index) {
// CHECK:           %[[VAL_2:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_3:.*]] = hlfir.eval_in_mem shape %[[VAL_2]] : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
// CHECK:           ^bb0(%[[VAL_4:.*]]: !fir.ref<!fir.array<?xf32>>):
// CHECK:             %[[VAL_5:.*]] = fir.call @_QPdyn_array_func(%[[VAL_1]]) : (index) -> !fir.array<?xf32>
// CHECK:             fir.save_result %[[VAL_5]] to %[[VAL_4]](%[[VAL_2]]) : !fir.array<?xf32>, !fir.ref<!fir.array<?xf32>>, !fir.shape<1>
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_3]] to %[[VAL_0]] : !hlfir.expr<?xf32>, !fir.box<!fir.array<?xf32>>
// CHECK:           hlfir.destroy %[[VAL_3]] : !hlfir.expr<?xf32>
// CHECK:           return
// CHECK:         }
// CHECK:         func.func private @_QPdyn_array_func(index) -> !fir.array<?xf32>
