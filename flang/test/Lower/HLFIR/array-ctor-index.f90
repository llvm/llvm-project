! Check that the implied-do index value is converted to proper type.
! RUN: bbc -emit-fir -hlfir -o - %s | FileCheck %s

function test1(k)
  integer*1 :: k
  integer*1 :: test1(4)
  test1 = ([(i*k, integer(8)::i=1,4)])
end function test1
! CHECK-LABEL:   func.func @_QPtest1(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<i8> {fir.bindc_name = "k"}) -> !fir.array<4xi8> {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtest1Ek"} : (!fir.ref<i8>) -> (!fir.ref<i8>, !fir.ref<i8>)
! CHECK:           %[[VAL_2:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.array<4xi8> {bindc_name = "test1", uniq_name = "_QFtest1Etest1"}
! CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_3]](%[[VAL_4]]) {uniq_name = "_QFtest1Etest1"} : (!fir.ref<!fir.array<4xi8>>, !fir.shape<1>) -> (!fir.ref<!fir.array<4xi8>>, !fir.ref<!fir.array<4xi8>>)
! CHECK:           %[[VAL_6:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_8:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i64) -> index
! CHECK:           %[[VAL_10:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i64) -> index
! CHECK:           %[[VAL_12:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_13:.*]] = hlfir.elemental %[[VAL_7]] : (!fir.shape<1>) -> !hlfir.expr<4xi64> {
! CHECK:           ^bb0(%[[VAL_14:.*]]: index):
! CHECK:             %[[VAL_15:.*]] = arith.subi %[[VAL_14]], %[[VAL_12]] : index
! CHECK:             %[[VAL_16:.*]] = arith.muli %[[VAL_15]], %[[VAL_11]] : index
! CHECK:             %[[VAL_17:.*]] = arith.addi %[[VAL_9]], %[[VAL_16]] : index
! CHECK:             %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (index) -> i64
! CHECK:             %[[VAL_19:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<i8>
! CHECK:             %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i8) -> i64
! CHECK:             %[[VAL_21:.*]] = arith.muli %[[VAL_18]], %[[VAL_20]] : i64
! CHECK:             hlfir.yield_element %[[VAL_21]] : i64
! CHECK:           }
! CHECK:           %[[VAL_22:.*]] = hlfir.elemental %[[VAL_7]] : (!fir.shape<1>) -> !hlfir.expr<4xi64> {
! CHECK:           ^bb0(%[[VAL_23:.*]]: index):
! CHECK:             %[[VAL_24:.*]] = hlfir.apply %[[VAL_25:.*]], %[[VAL_23]] : (!hlfir.expr<4xi64>, index) -> i64
! CHECK:             %[[VAL_26:.*]] = hlfir.no_reassoc %[[VAL_24]] : i64
! CHECK:             hlfir.yield_element %[[VAL_26]] : i64
! CHECK:           }
! CHECK:           %[[VAL_27:.*]] = hlfir.elemental %[[VAL_7]] : (!fir.shape<1>) -> !hlfir.expr<4xi8> {
! CHECK:           ^bb0(%[[VAL_28:.*]]: index):
! CHECK:             %[[VAL_29:.*]] = hlfir.apply %[[VAL_30:.*]], %[[VAL_28]] : (!hlfir.expr<4xi64>, index) -> i64
! CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_29]] : (i64) -> i8
! CHECK:             hlfir.yield_element %[[VAL_31]] : i8
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_32:.*]] to %[[VAL_5]]#0 : !hlfir.expr<4xi8>, !fir.ref<!fir.array<4xi8>>
! CHECK:           hlfir.destroy %[[VAL_32]] : !hlfir.expr<4xi8>
! CHECK:           hlfir.destroy %[[VAL_33:.*]] : !hlfir.expr<4xi64>
! CHECK:           hlfir.destroy %[[VAL_34:.*]] : !hlfir.expr<4xi64>
! CHECK:           %[[VAL_35:.*]] = fir.load %[[VAL_5]]#1 : !fir.ref<!fir.array<4xi8>>
! CHECK:           return %[[VAL_35]] : !fir.array<4xi8>
! CHECK:         }

function test2(k)
  integer*2 :: k
  integer*2 :: test2(4)
  test2 = ([(i*k, integer(8)::i=1,4)])
end function test2
! CHECK-LABEL:   func.func @_QPtest2(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<i16> {fir.bindc_name = "k"}) -> !fir.array<4xi16> {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtest2Ek"} : (!fir.ref<i16>) -> (!fir.ref<i16>, !fir.ref<i16>)
! CHECK:           %[[VAL_2:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.array<4xi16> {bindc_name = "test2", uniq_name = "_QFtest2Etest2"}
! CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_3]](%[[VAL_4]]) {uniq_name = "_QFtest2Etest2"} : (!fir.ref<!fir.array<4xi16>>, !fir.shape<1>) -> (!fir.ref<!fir.array<4xi16>>, !fir.ref<!fir.array<4xi16>>)
! CHECK:           %[[VAL_6:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_8:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i64) -> index
! CHECK:           %[[VAL_10:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i64) -> index
! CHECK:           %[[VAL_12:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_13:.*]] = hlfir.elemental %[[VAL_7]] : (!fir.shape<1>) -> !hlfir.expr<4xi64> {
! CHECK:           ^bb0(%[[VAL_14:.*]]: index):
! CHECK:             %[[VAL_15:.*]] = arith.subi %[[VAL_14]], %[[VAL_12]] : index
! CHECK:             %[[VAL_16:.*]] = arith.muli %[[VAL_15]], %[[VAL_11]] : index
! CHECK:             %[[VAL_17:.*]] = arith.addi %[[VAL_9]], %[[VAL_16]] : index
! CHECK:             %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (index) -> i64
! CHECK:             %[[VAL_19:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<i16>
! CHECK:             %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i16) -> i64
! CHECK:             %[[VAL_21:.*]] = arith.muli %[[VAL_18]], %[[VAL_20]] : i64
! CHECK:             hlfir.yield_element %[[VAL_21]] : i64
! CHECK:           }
! CHECK:           %[[VAL_22:.*]] = hlfir.elemental %[[VAL_7]] : (!fir.shape<1>) -> !hlfir.expr<4xi64> {
! CHECK:           ^bb0(%[[VAL_23:.*]]: index):
! CHECK:             %[[VAL_24:.*]] = hlfir.apply %[[VAL_25:.*]], %[[VAL_23]] : (!hlfir.expr<4xi64>, index) -> i64
! CHECK:             %[[VAL_26:.*]] = hlfir.no_reassoc %[[VAL_24]] : i64
! CHECK:             hlfir.yield_element %[[VAL_26]] : i64
! CHECK:           }
! CHECK:           %[[VAL_27:.*]] = hlfir.elemental %[[VAL_7]] : (!fir.shape<1>) -> !hlfir.expr<4xi16> {
! CHECK:           ^bb0(%[[VAL_28:.*]]: index):
! CHECK:             %[[VAL_29:.*]] = hlfir.apply %[[VAL_30:.*]], %[[VAL_28]] : (!hlfir.expr<4xi64>, index) -> i64
! CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_29]] : (i64) -> i16
! CHECK:             hlfir.yield_element %[[VAL_31]] : i16
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_32:.*]] to %[[VAL_5]]#0 : !hlfir.expr<4xi16>, !fir.ref<!fir.array<4xi16>>
! CHECK:           hlfir.destroy %[[VAL_32]] : !hlfir.expr<4xi16>
! CHECK:           hlfir.destroy %[[VAL_33:.*]] : !hlfir.expr<4xi64>
! CHECK:           hlfir.destroy %[[VAL_34:.*]] : !hlfir.expr<4xi64>
! CHECK:           %[[VAL_35:.*]] = fir.load %[[VAL_5]]#1 : !fir.ref<!fir.array<4xi16>>
! CHECK:           return %[[VAL_35]] : !fir.array<4xi16>
! CHECK:         }

function test3(k)
  integer*4 :: k
  integer*4 :: test3(4)
  test3 = ([(i*k, integer(8)::i=1,4)])
end function test3
! CHECK-LABEL:   func.func @_QPtest3(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "k"}) -> !fir.array<4xi32> {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtest3Ek"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_2:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.array<4xi32> {bindc_name = "test3", uniq_name = "_QFtest3Etest3"}
! CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_3]](%[[VAL_4]]) {uniq_name = "_QFtest3Etest3"} : (!fir.ref<!fir.array<4xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<4xi32>>, !fir.ref<!fir.array<4xi32>>)
! CHECK:           %[[VAL_6:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_8:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i64) -> index
! CHECK:           %[[VAL_10:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i64) -> index
! CHECK:           %[[VAL_12:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_13:.*]] = hlfir.elemental %[[VAL_7]] : (!fir.shape<1>) -> !hlfir.expr<4xi64> {
! CHECK:           ^bb0(%[[VAL_14:.*]]: index):
! CHECK:             %[[VAL_15:.*]] = arith.subi %[[VAL_14]], %[[VAL_12]] : index
! CHECK:             %[[VAL_16:.*]] = arith.muli %[[VAL_15]], %[[VAL_11]] : index
! CHECK:             %[[VAL_17:.*]] = arith.addi %[[VAL_9]], %[[VAL_16]] : index
! CHECK:             %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (index) -> i64
! CHECK:             %[[VAL_19:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<i32>
! CHECK:             %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i32) -> i64
! CHECK:             %[[VAL_21:.*]] = arith.muli %[[VAL_18]], %[[VAL_20]] : i64
! CHECK:             hlfir.yield_element %[[VAL_21]] : i64
! CHECK:           }
! CHECK:           %[[VAL_22:.*]] = hlfir.elemental %[[VAL_7]] : (!fir.shape<1>) -> !hlfir.expr<4xi64> {
! CHECK:           ^bb0(%[[VAL_23:.*]]: index):
! CHECK:             %[[VAL_24:.*]] = hlfir.apply %[[VAL_25:.*]], %[[VAL_23]] : (!hlfir.expr<4xi64>, index) -> i64
! CHECK:             %[[VAL_26:.*]] = hlfir.no_reassoc %[[VAL_24]] : i64
! CHECK:             hlfir.yield_element %[[VAL_26]] : i64
! CHECK:           }
! CHECK:           %[[VAL_27:.*]] = hlfir.elemental %[[VAL_7]] : (!fir.shape<1>) -> !hlfir.expr<4xi32> {
! CHECK:           ^bb0(%[[VAL_28:.*]]: index):
! CHECK:             %[[VAL_29:.*]] = hlfir.apply %[[VAL_30:.*]], %[[VAL_28]] : (!hlfir.expr<4xi64>, index) -> i64
! CHECK:             %[[VAL_31:.*]] = fir.convert %[[VAL_29]] : (i64) -> i32
! CHECK:             hlfir.yield_element %[[VAL_31]] : i32
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_32:.*]] to %[[VAL_5]]#0 : !hlfir.expr<4xi32>, !fir.ref<!fir.array<4xi32>>
! CHECK:           hlfir.destroy %[[VAL_32]] : !hlfir.expr<4xi32>
! CHECK:           hlfir.destroy %[[VAL_33:.*]] : !hlfir.expr<4xi64>
! CHECK:           hlfir.destroy %[[VAL_34:.*]] : !hlfir.expr<4xi64>
! CHECK:           %[[VAL_35:.*]] = fir.load %[[VAL_5]]#1 : !fir.ref<!fir.array<4xi32>>
! CHECK:           return %[[VAL_35]] : !fir.array<4xi32>
! CHECK:         }

function test4(k)
  integer*8 :: k
  integer*8 :: test4(4)
  test4 = ([(i*k, integer(8)::i=1,4)])
end function test4
! CHECK-LABEL:   func.func @_QPtest4(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<i64> {fir.bindc_name = "k"}) -> !fir.array<4xi64> {
! CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtest4Ek"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK:           %[[VAL_2:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.array<4xi64> {bindc_name = "test4", uniq_name = "_QFtest4Etest4"}
! CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_3]](%[[VAL_4]]) {uniq_name = "_QFtest4Etest4"} : (!fir.ref<!fir.array<4xi64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<4xi64>>, !fir.ref<!fir.array<4xi64>>)
! CHECK:           %[[VAL_6:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_7:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_8:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i64) -> index
! CHECK:           %[[VAL_10:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i64) -> index
! CHECK:           %[[VAL_12:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_13:.*]] = hlfir.elemental %[[VAL_7]] : (!fir.shape<1>) -> !hlfir.expr<4xi64> {
! CHECK:           ^bb0(%[[VAL_14:.*]]: index):
! CHECK:             %[[VAL_15:.*]] = arith.subi %[[VAL_14]], %[[VAL_12]] : index
! CHECK:             %[[VAL_16:.*]] = arith.muli %[[VAL_15]], %[[VAL_11]] : index
! CHECK:             %[[VAL_17:.*]] = arith.addi %[[VAL_9]], %[[VAL_16]] : index
! CHECK:             %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (index) -> i64
! CHECK:             %[[VAL_19:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<i64>
! CHECK:             %[[VAL_20:.*]] = arith.muli %[[VAL_18]], %[[VAL_19]] : i64
! CHECK:             hlfir.yield_element %[[VAL_20]] : i64
! CHECK:           }
! CHECK:           %[[VAL_21:.*]] = hlfir.elemental %[[VAL_7]] : (!fir.shape<1>) -> !hlfir.expr<4xi64> {
! CHECK:           ^bb0(%[[VAL_22:.*]]: index):
! CHECK:             %[[VAL_23:.*]] = hlfir.apply %[[VAL_24:.*]], %[[VAL_22]] : (!hlfir.expr<4xi64>, index) -> i64
! CHECK:             %[[VAL_25:.*]] = hlfir.no_reassoc %[[VAL_23]] : i64
! CHECK:             hlfir.yield_element %[[VAL_25]] : i64
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_26:.*]] to %[[VAL_5]]#0 : !hlfir.expr<4xi64>, !fir.ref<!fir.array<4xi64>>
! CHECK:           hlfir.destroy %[[VAL_26]] : !hlfir.expr<4xi64>
! CHECK:           hlfir.destroy %[[VAL_27:.*]] : !hlfir.expr<4xi64>
! CHECK:           %[[VAL_28:.*]] = fir.load %[[VAL_5]]#1 : !fir.ref<!fir.array<4xi64>>
! CHECK:           return %[[VAL_28]] : !fir.array<4xi64>
! CHECK:         }
