! Check that the implied-do index value is converted to proper type.
! RUN: bbc -emit-fir -hlfir=false -o - %s | FileCheck %s

function test1(k)
  integer*1 :: k
  integer*1 :: test1(4)
  test1 = ([(i*k, integer(8)::i=1,4)])
end function test1
! CHECK-LABEL:   func.func @_QPtest1(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<i8> {fir.bindc_name = "k"}) -> !fir.array<4xi8> {
! CHECK:           %[[VAL_1:.*]] = fir.alloca index {bindc_name = ".buff.pos"}
! CHECK:           %[[VAL_2:.*]] = fir.alloca index {bindc_name = ".buff.size"}
! CHECK:           %[[VAL_3:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_4:.*]] = fir.alloca !fir.array<4xi8> {bindc_name = "test1", uniq_name = "_QFtest1Etest1"}
! CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_6:.*]] = fir.array_load %[[VAL_4]](%[[VAL_5]]) : (!fir.ref<!fir.array<4xi8>>, !fir.shape<1>) -> !fir.array<4xi8>
! CHECK:           %[[VAL_7:.*]] = arith.constant 0 : index
! CHECK:           fir.store %[[VAL_7]] to %[[VAL_1]] : !fir.ref<index>
! CHECK:           %[[VAL_8:.*]] = fir.allocmem !fir.array<4xi64>
! CHECK:           %[[VAL_9:.*]] = arith.constant 4 : index
! CHECK:           fir.store %[[VAL_9]] to %[[VAL_2]] : !fir.ref<index>
! CHECK:           %[[VAL_10:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i64) -> index
! CHECK:           %[[VAL_12:.*]] = arith.constant 4 : i64
! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i64) -> index
! CHECK:           %[[VAL_14:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (i64) -> index
! CHECK:           %[[VAL_16:.*]] = fir.do_loop %[[VAL_17:.*]] = %[[VAL_11]] to %[[VAL_13]] step %[[VAL_15]] iter_args(%[[VAL_18:.*]] = %[[VAL_8]]) -> (!fir.heap<!fir.array<4xi64>>) {
! CHECK:             %[[VAL_19:.*]] = fir.convert %[[VAL_17]] : (index) -> i64
! CHECK:             %[[VAL_20:.*]] = fir.load %[[VAL_0]] : !fir.ref<i8>
! CHECK:             %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i8) -> i64
! CHECK:             %[[VAL_22:.*]] = arith.muli %[[VAL_19]], %[[VAL_21]] : i64
! CHECK:             %[[VAL_23:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_24:.*]] = fir.zero_bits !fir.ref<!fir.array<4xi64>>
! CHECK:             %[[VAL_25:.*]] = fir.coordinate_of %[[VAL_24]], %[[VAL_23]] : (!fir.ref<!fir.array<4xi64>>, index) -> !fir.ref<i64>
! CHECK:             %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (!fir.ref<i64>) -> index
! CHECK:             %[[VAL_27:.*]] = fir.load %[[VAL_1]] : !fir.ref<index>
! CHECK:             %[[VAL_28:.*]] = fir.load %[[VAL_2]] : !fir.ref<index>
! CHECK:             %[[VAL_29:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_30:.*]] = arith.addi %[[VAL_27]], %[[VAL_29]] : index
! CHECK:             %[[VAL_31:.*]] = arith.cmpi sle, %[[VAL_28]], %[[VAL_30]] : index
! CHECK:             %[[VAL_32:.*]] = fir.if %[[VAL_31]] -> (!fir.heap<!fir.array<4xi64>>) {
! CHECK:               %[[VAL_33:.*]] = arith.constant 2 : index
! CHECK:               %[[VAL_34:.*]] = arith.muli %[[VAL_30]], %[[VAL_33]] : index
! CHECK:               fir.store %[[VAL_34]] to %[[VAL_2]] : !fir.ref<index>
! CHECK:               %[[VAL_35:.*]] = arith.muli %[[VAL_34]], %[[VAL_26]] : index
! CHECK:               %[[VAL_36:.*]] = fir.convert %[[VAL_18]] : (!fir.heap<!fir.array<4xi64>>) -> !fir.ref<i8>
! CHECK:               %[[VAL_37:.*]] = fir.convert %[[VAL_35]] : (index) -> i64
! CHECK:               %[[VAL_38:.*]] = fir.call @realloc(%[[VAL_36]], %[[VAL_37]]) fastmath<contract> : (!fir.ref<i8>, i64) -> !fir.ref<i8>
! CHECK:               %[[VAL_39:.*]] = fir.convert %[[VAL_38]] : (!fir.ref<i8>) -> !fir.heap<!fir.array<4xi64>>
! CHECK:               fir.result %[[VAL_39]] : !fir.heap<!fir.array<4xi64>>
! CHECK:             } else {
! CHECK:               fir.result %[[VAL_18]] : !fir.heap<!fir.array<4xi64>>
! CHECK:             }
! CHECK:             %[[VAL_40:.*]] = fir.coordinate_of %[[VAL_41:.*]], %[[VAL_27]] : (!fir.heap<!fir.array<4xi64>>, index) -> !fir.ref<i64>
! CHECK:             fir.store %[[VAL_22]] to %[[VAL_40]] : !fir.ref<i64>
! CHECK:             fir.store %[[VAL_30]] to %[[VAL_1]] : !fir.ref<index>
! CHECK:             fir.result %[[VAL_41]] : !fir.heap<!fir.array<4xi64>>
! CHECK:           }
! CHECK:           %[[VAL_42:.*]] = fir.load %[[VAL_1]] : !fir.ref<index>
! CHECK:           %[[VAL_43:.*]] = fir.shape %[[VAL_42]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_44:.*]] = fir.array_load %[[VAL_45:.*]](%[[VAL_43]]) : (!fir.heap<!fir.array<4xi64>>, !fir.shape<1>) -> !fir.array<4xi64>
! CHECK:           %[[VAL_46:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_47:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_48:.*]] = arith.subi %[[VAL_3]], %[[VAL_46]] : index
! CHECK:           %[[VAL_49:.*]] = fir.do_loop %[[VAL_50:.*]] = %[[VAL_47]] to %[[VAL_48]] step %[[VAL_46]] unordered iter_args(%[[VAL_51:.*]] = %[[VAL_6]]) -> (!fir.array<4xi8>) {
! CHECK:             %[[VAL_52:.*]] = fir.array_fetch %[[VAL_44]], %[[VAL_50]] : (!fir.array<4xi64>, index) -> i64
! CHECK:             %[[VAL_53:.*]] = fir.no_reassoc %[[VAL_52]] : i64
! CHECK:             %[[VAL_54:.*]] = fir.convert %[[VAL_53]] : (i64) -> i8
! CHECK:             %[[VAL_55:.*]] = fir.array_update %[[VAL_51]], %[[VAL_54]], %[[VAL_50]] : (!fir.array<4xi8>, i8, index) -> !fir.array<4xi8>
! CHECK:             fir.result %[[VAL_55]] : !fir.array<4xi8>
! CHECK:           }
! CHECK:           fir.array_merge_store %[[VAL_6]], %[[VAL_56:.*]] to %[[VAL_4]] : !fir.array<4xi8>, !fir.array<4xi8>, !fir.ref<!fir.array<4xi8>>
! CHECK:           fir.freemem %[[VAL_45]] : !fir.heap<!fir.array<4xi64>>
! CHECK:           %[[VAL_57:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.array<4xi8>>
! CHECK:           return %[[VAL_57]] : !fir.array<4xi8>
! CHECK:         }

function test2(k)
  integer*2 :: k
  integer*2 :: test2(4)
  test2 = ([(i*k, integer(8)::i=1,4)])
end function test2
! CHECK-LABEL:   func.func @_QPtest2(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<i16> {fir.bindc_name = "k"}) -> !fir.array<4xi16> {
! CHECK:           %[[VAL_1:.*]] = fir.alloca index {bindc_name = ".buff.pos"}
! CHECK:           %[[VAL_2:.*]] = fir.alloca index {bindc_name = ".buff.size"}
! CHECK:           %[[VAL_3:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_4:.*]] = fir.alloca !fir.array<4xi16> {bindc_name = "test2", uniq_name = "_QFtest2Etest2"}
! CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_6:.*]] = fir.array_load %[[VAL_4]](%[[VAL_5]]) : (!fir.ref<!fir.array<4xi16>>, !fir.shape<1>) -> !fir.array<4xi16>
! CHECK:           %[[VAL_7:.*]] = arith.constant 0 : index
! CHECK:           fir.store %[[VAL_7]] to %[[VAL_1]] : !fir.ref<index>
! CHECK:           %[[VAL_8:.*]] = fir.allocmem !fir.array<4xi64>
! CHECK:           %[[VAL_9:.*]] = arith.constant 4 : index
! CHECK:           fir.store %[[VAL_9]] to %[[VAL_2]] : !fir.ref<index>
! CHECK:           %[[VAL_10:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i64) -> index
! CHECK:           %[[VAL_12:.*]] = arith.constant 4 : i64
! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i64) -> index
! CHECK:           %[[VAL_14:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (i64) -> index
! CHECK:           %[[VAL_16:.*]] = fir.do_loop %[[VAL_17:.*]] = %[[VAL_11]] to %[[VAL_13]] step %[[VAL_15]] iter_args(%[[VAL_18:.*]] = %[[VAL_8]]) -> (!fir.heap<!fir.array<4xi64>>) {
! CHECK:             %[[VAL_19:.*]] = fir.convert %[[VAL_17]] : (index) -> i64
! CHECK:             %[[VAL_20:.*]] = fir.load %[[VAL_0]] : !fir.ref<i16>
! CHECK:             %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i16) -> i64
! CHECK:             %[[VAL_22:.*]] = arith.muli %[[VAL_19]], %[[VAL_21]] : i64
! CHECK:             %[[VAL_23:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_24:.*]] = fir.zero_bits !fir.ref<!fir.array<4xi64>>
! CHECK:             %[[VAL_25:.*]] = fir.coordinate_of %[[VAL_24]], %[[VAL_23]] : (!fir.ref<!fir.array<4xi64>>, index) -> !fir.ref<i64>
! CHECK:             %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (!fir.ref<i64>) -> index
! CHECK:             %[[VAL_27:.*]] = fir.load %[[VAL_1]] : !fir.ref<index>
! CHECK:             %[[VAL_28:.*]] = fir.load %[[VAL_2]] : !fir.ref<index>
! CHECK:             %[[VAL_29:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_30:.*]] = arith.addi %[[VAL_27]], %[[VAL_29]] : index
! CHECK:             %[[VAL_31:.*]] = arith.cmpi sle, %[[VAL_28]], %[[VAL_30]] : index
! CHECK:             %[[VAL_32:.*]] = fir.if %[[VAL_31]] -> (!fir.heap<!fir.array<4xi64>>) {
! CHECK:               %[[VAL_33:.*]] = arith.constant 2 : index
! CHECK:               %[[VAL_34:.*]] = arith.muli %[[VAL_30]], %[[VAL_33]] : index
! CHECK:               fir.store %[[VAL_34]] to %[[VAL_2]] : !fir.ref<index>
! CHECK:               %[[VAL_35:.*]] = arith.muli %[[VAL_34]], %[[VAL_26]] : index
! CHECK:               %[[VAL_36:.*]] = fir.convert %[[VAL_18]] : (!fir.heap<!fir.array<4xi64>>) -> !fir.ref<i8>
! CHECK:               %[[VAL_37:.*]] = fir.convert %[[VAL_35]] : (index) -> i64
! CHECK:               %[[VAL_38:.*]] = fir.call @realloc(%[[VAL_36]], %[[VAL_37]]) fastmath<contract> : (!fir.ref<i8>, i64) -> !fir.ref<i8>
! CHECK:               %[[VAL_39:.*]] = fir.convert %[[VAL_38]] : (!fir.ref<i8>) -> !fir.heap<!fir.array<4xi64>>
! CHECK:               fir.result %[[VAL_39]] : !fir.heap<!fir.array<4xi64>>
! CHECK:             } else {
! CHECK:               fir.result %[[VAL_18]] : !fir.heap<!fir.array<4xi64>>
! CHECK:             }
! CHECK:             %[[VAL_40:.*]] = fir.coordinate_of %[[VAL_41:.*]], %[[VAL_27]] : (!fir.heap<!fir.array<4xi64>>, index) -> !fir.ref<i64>
! CHECK:             fir.store %[[VAL_22]] to %[[VAL_40]] : !fir.ref<i64>
! CHECK:             fir.store %[[VAL_30]] to %[[VAL_1]] : !fir.ref<index>
! CHECK:             fir.result %[[VAL_41]] : !fir.heap<!fir.array<4xi64>>
! CHECK:           }
! CHECK:           %[[VAL_42:.*]] = fir.load %[[VAL_1]] : !fir.ref<index>
! CHECK:           %[[VAL_43:.*]] = fir.shape %[[VAL_42]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_44:.*]] = fir.array_load %[[VAL_45:.*]](%[[VAL_43]]) : (!fir.heap<!fir.array<4xi64>>, !fir.shape<1>) -> !fir.array<4xi64>
! CHECK:           %[[VAL_46:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_47:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_48:.*]] = arith.subi %[[VAL_3]], %[[VAL_46]] : index
! CHECK:           %[[VAL_49:.*]] = fir.do_loop %[[VAL_50:.*]] = %[[VAL_47]] to %[[VAL_48]] step %[[VAL_46]] unordered iter_args(%[[VAL_51:.*]] = %[[VAL_6]]) -> (!fir.array<4xi16>) {
! CHECK:             %[[VAL_52:.*]] = fir.array_fetch %[[VAL_44]], %[[VAL_50]] : (!fir.array<4xi64>, index) -> i64
! CHECK:             %[[VAL_53:.*]] = fir.no_reassoc %[[VAL_52]] : i64
! CHECK:             %[[VAL_54:.*]] = fir.convert %[[VAL_53]] : (i64) -> i16
! CHECK:             %[[VAL_55:.*]] = fir.array_update %[[VAL_51]], %[[VAL_54]], %[[VAL_50]] : (!fir.array<4xi16>, i16, index) -> !fir.array<4xi16>
! CHECK:             fir.result %[[VAL_55]] : !fir.array<4xi16>
! CHECK:           }
! CHECK:           fir.array_merge_store %[[VAL_6]], %[[VAL_56:.*]] to %[[VAL_4]] : !fir.array<4xi16>, !fir.array<4xi16>, !fir.ref<!fir.array<4xi16>>
! CHECK:           fir.freemem %[[VAL_45]] : !fir.heap<!fir.array<4xi64>>
! CHECK:           %[[VAL_57:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.array<4xi16>>
! CHECK:           return %[[VAL_57]] : !fir.array<4xi16>
! CHECK:         }

function test3(k)
  integer*4 :: k
  integer*4 :: test3(4)
  test3 = ([(i*k, integer(8)::i=1,4)])
end function test3
! CHECK-LABEL:   func.func @_QPtest3(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "k"}) -> !fir.array<4xi32> {
! CHECK:           %[[VAL_1:.*]] = fir.alloca index {bindc_name = ".buff.pos"}
! CHECK:           %[[VAL_2:.*]] = fir.alloca index {bindc_name = ".buff.size"}
! CHECK:           %[[VAL_3:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_4:.*]] = fir.alloca !fir.array<4xi32> {bindc_name = "test3", uniq_name = "_QFtest3Etest3"}
! CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_6:.*]] = fir.array_load %[[VAL_4]](%[[VAL_5]]) : (!fir.ref<!fir.array<4xi32>>, !fir.shape<1>) -> !fir.array<4xi32>
! CHECK:           %[[VAL_7:.*]] = arith.constant 0 : index
! CHECK:           fir.store %[[VAL_7]] to %[[VAL_1]] : !fir.ref<index>
! CHECK:           %[[VAL_8:.*]] = fir.allocmem !fir.array<4xi64>
! CHECK:           %[[VAL_9:.*]] = arith.constant 4 : index
! CHECK:           fir.store %[[VAL_9]] to %[[VAL_2]] : !fir.ref<index>
! CHECK:           %[[VAL_10:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i64) -> index
! CHECK:           %[[VAL_12:.*]] = arith.constant 4 : i64
! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i64) -> index
! CHECK:           %[[VAL_14:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (i64) -> index
! CHECK:           %[[VAL_16:.*]] = fir.do_loop %[[VAL_17:.*]] = %[[VAL_11]] to %[[VAL_13]] step %[[VAL_15]] iter_args(%[[VAL_18:.*]] = %[[VAL_8]]) -> (!fir.heap<!fir.array<4xi64>>) {
! CHECK:             %[[VAL_19:.*]] = fir.convert %[[VAL_17]] : (index) -> i64
! CHECK:             %[[VAL_20:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:             %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i32) -> i64
! CHECK:             %[[VAL_22:.*]] = arith.muli %[[VAL_19]], %[[VAL_21]] : i64
! CHECK:             %[[VAL_23:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_24:.*]] = fir.zero_bits !fir.ref<!fir.array<4xi64>>
! CHECK:             %[[VAL_25:.*]] = fir.coordinate_of %[[VAL_24]], %[[VAL_23]] : (!fir.ref<!fir.array<4xi64>>, index) -> !fir.ref<i64>
! CHECK:             %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (!fir.ref<i64>) -> index
! CHECK:             %[[VAL_27:.*]] = fir.load %[[VAL_1]] : !fir.ref<index>
! CHECK:             %[[VAL_28:.*]] = fir.load %[[VAL_2]] : !fir.ref<index>
! CHECK:             %[[VAL_29:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_30:.*]] = arith.addi %[[VAL_27]], %[[VAL_29]] : index
! CHECK:             %[[VAL_31:.*]] = arith.cmpi sle, %[[VAL_28]], %[[VAL_30]] : index
! CHECK:             %[[VAL_32:.*]] = fir.if %[[VAL_31]] -> (!fir.heap<!fir.array<4xi64>>) {
! CHECK:               %[[VAL_33:.*]] = arith.constant 2 : index
! CHECK:               %[[VAL_34:.*]] = arith.muli %[[VAL_30]], %[[VAL_33]] : index
! CHECK:               fir.store %[[VAL_34]] to %[[VAL_2]] : !fir.ref<index>
! CHECK:               %[[VAL_35:.*]] = arith.muli %[[VAL_34]], %[[VAL_26]] : index
! CHECK:               %[[VAL_36:.*]] = fir.convert %[[VAL_18]] : (!fir.heap<!fir.array<4xi64>>) -> !fir.ref<i8>
! CHECK:               %[[VAL_37:.*]] = fir.convert %[[VAL_35]] : (index) -> i64
! CHECK:               %[[VAL_38:.*]] = fir.call @realloc(%[[VAL_36]], %[[VAL_37]]) fastmath<contract> : (!fir.ref<i8>, i64) -> !fir.ref<i8>
! CHECK:               %[[VAL_39:.*]] = fir.convert %[[VAL_38]] : (!fir.ref<i8>) -> !fir.heap<!fir.array<4xi64>>
! CHECK:               fir.result %[[VAL_39]] : !fir.heap<!fir.array<4xi64>>
! CHECK:             } else {
! CHECK:               fir.result %[[VAL_18]] : !fir.heap<!fir.array<4xi64>>
! CHECK:             }
! CHECK:             %[[VAL_40:.*]] = fir.coordinate_of %[[VAL_41:.*]], %[[VAL_27]] : (!fir.heap<!fir.array<4xi64>>, index) -> !fir.ref<i64>
! CHECK:             fir.store %[[VAL_22]] to %[[VAL_40]] : !fir.ref<i64>
! CHECK:             fir.store %[[VAL_30]] to %[[VAL_1]] : !fir.ref<index>
! CHECK:             fir.result %[[VAL_41]] : !fir.heap<!fir.array<4xi64>>
! CHECK:           }
! CHECK:           %[[VAL_42:.*]] = fir.load %[[VAL_1]] : !fir.ref<index>
! CHECK:           %[[VAL_43:.*]] = fir.shape %[[VAL_42]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_44:.*]] = fir.array_load %[[VAL_45:.*]](%[[VAL_43]]) : (!fir.heap<!fir.array<4xi64>>, !fir.shape<1>) -> !fir.array<4xi64>
! CHECK:           %[[VAL_46:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_47:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_48:.*]] = arith.subi %[[VAL_3]], %[[VAL_46]] : index
! CHECK:           %[[VAL_49:.*]] = fir.do_loop %[[VAL_50:.*]] = %[[VAL_47]] to %[[VAL_48]] step %[[VAL_46]] unordered iter_args(%[[VAL_51:.*]] = %[[VAL_6]]) -> (!fir.array<4xi32>) {
! CHECK:             %[[VAL_52:.*]] = fir.array_fetch %[[VAL_44]], %[[VAL_50]] : (!fir.array<4xi64>, index) -> i64
! CHECK:             %[[VAL_53:.*]] = fir.no_reassoc %[[VAL_52]] : i64
! CHECK:             %[[VAL_54:.*]] = fir.convert %[[VAL_53]] : (i64) -> i32
! CHECK:             %[[VAL_55:.*]] = fir.array_update %[[VAL_51]], %[[VAL_54]], %[[VAL_50]] : (!fir.array<4xi32>, i32, index) -> !fir.array<4xi32>
! CHECK:             fir.result %[[VAL_55]] : !fir.array<4xi32>
! CHECK:           }
! CHECK:           fir.array_merge_store %[[VAL_6]], %[[VAL_56:.*]] to %[[VAL_4]] : !fir.array<4xi32>, !fir.array<4xi32>, !fir.ref<!fir.array<4xi32>>
! CHECK:           fir.freemem %[[VAL_45]] : !fir.heap<!fir.array<4xi64>>
! CHECK:           %[[VAL_57:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.array<4xi32>>
! CHECK:           return %[[VAL_57]] : !fir.array<4xi32>
! CHECK:         }

function test4(k)
  integer*8 :: k
  integer*8 :: test4(4)
  test4 = ([(i*k, integer(8)::i=1,4)])
end function test4
! CHECK-LABEL:   func.func @_QPtest4(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<i64> {fir.bindc_name = "k"}) -> !fir.array<4xi64> {
! CHECK:           %[[VAL_1:.*]] = fir.alloca index {bindc_name = ".buff.pos"}
! CHECK:           %[[VAL_2:.*]] = fir.alloca index {bindc_name = ".buff.size"}
! CHECK:           %[[VAL_3:.*]] = arith.constant 4 : index
! CHECK:           %[[VAL_4:.*]] = fir.alloca !fir.array<4xi64> {bindc_name = "test4", uniq_name = "_QFtest4Etest4"}
! CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_6:.*]] = fir.array_load %[[VAL_4]](%[[VAL_5]]) : (!fir.ref<!fir.array<4xi64>>, !fir.shape<1>) -> !fir.array<4xi64>
! CHECK:           %[[VAL_7:.*]] = arith.constant 0 : index
! CHECK:           fir.store %[[VAL_7]] to %[[VAL_1]] : !fir.ref<index>
! CHECK:           %[[VAL_8:.*]] = fir.allocmem !fir.array<4xi64>
! CHECK:           %[[VAL_9:.*]] = arith.constant 4 : index
! CHECK:           fir.store %[[VAL_9]] to %[[VAL_2]] : !fir.ref<index>
! CHECK:           %[[VAL_10:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i64) -> index
! CHECK:           %[[VAL_12:.*]] = arith.constant 4 : i64
! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i64) -> index
! CHECK:           %[[VAL_14:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_14]] : (i64) -> index
! CHECK:           %[[VAL_16:.*]] = fir.do_loop %[[VAL_17:.*]] = %[[VAL_11]] to %[[VAL_13]] step %[[VAL_15]] iter_args(%[[VAL_18:.*]] = %[[VAL_8]]) -> (!fir.heap<!fir.array<4xi64>>) {
! CHECK:             %[[VAL_19:.*]] = fir.convert %[[VAL_17]] : (index) -> i64
! CHECK:             %[[VAL_20:.*]] = fir.load %[[VAL_0]] : !fir.ref<i64>
! CHECK:             %[[VAL_21:.*]] = arith.muli %[[VAL_19]], %[[VAL_20]] : i64
! CHECK:             %[[VAL_22:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_23:.*]] = fir.zero_bits !fir.ref<!fir.array<4xi64>>
! CHECK:             %[[VAL_24:.*]] = fir.coordinate_of %[[VAL_23]], %[[VAL_22]] : (!fir.ref<!fir.array<4xi64>>, index) -> !fir.ref<i64>
! CHECK:             %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (!fir.ref<i64>) -> index
! CHECK:             %[[VAL_26:.*]] = fir.load %[[VAL_1]] : !fir.ref<index>
! CHECK:             %[[VAL_27:.*]] = fir.load %[[VAL_2]] : !fir.ref<index>
! CHECK:             %[[VAL_28:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_29:.*]] = arith.addi %[[VAL_26]], %[[VAL_28]] : index
! CHECK:             %[[VAL_30:.*]] = arith.cmpi sle, %[[VAL_27]], %[[VAL_29]] : index
! CHECK:             %[[VAL_31:.*]] = fir.if %[[VAL_30]] -> (!fir.heap<!fir.array<4xi64>>) {
! CHECK:               %[[VAL_32:.*]] = arith.constant 2 : index
! CHECK:               %[[VAL_33:.*]] = arith.muli %[[VAL_29]], %[[VAL_32]] : index
! CHECK:               fir.store %[[VAL_33]] to %[[VAL_2]] : !fir.ref<index>
! CHECK:               %[[VAL_34:.*]] = arith.muli %[[VAL_33]], %[[VAL_25]] : index
! CHECK:               %[[VAL_35:.*]] = fir.convert %[[VAL_18]] : (!fir.heap<!fir.array<4xi64>>) -> !fir.ref<i8>
! CHECK:               %[[VAL_36:.*]] = fir.convert %[[VAL_34]] : (index) -> i64
! CHECK:               %[[VAL_37:.*]] = fir.call @realloc(%[[VAL_35]], %[[VAL_36]]) fastmath<contract> : (!fir.ref<i8>, i64) -> !fir.ref<i8>
! CHECK:               %[[VAL_38:.*]] = fir.convert %[[VAL_37]] : (!fir.ref<i8>) -> !fir.heap<!fir.array<4xi64>>
! CHECK:               fir.result %[[VAL_38]] : !fir.heap<!fir.array<4xi64>>
! CHECK:             } else {
! CHECK:               fir.result %[[VAL_18]] : !fir.heap<!fir.array<4xi64>>
! CHECK:             }
! CHECK:             %[[VAL_39:.*]] = fir.coordinate_of %[[VAL_40:.*]], %[[VAL_26]] : (!fir.heap<!fir.array<4xi64>>, index) -> !fir.ref<i64>
! CHECK:             fir.store %[[VAL_21]] to %[[VAL_39]] : !fir.ref<i64>
! CHECK:             fir.store %[[VAL_29]] to %[[VAL_1]] : !fir.ref<index>
! CHECK:             fir.result %[[VAL_40]] : !fir.heap<!fir.array<4xi64>>
! CHECK:           }
! CHECK:           %[[VAL_41:.*]] = fir.load %[[VAL_1]] : !fir.ref<index>
! CHECK:           %[[VAL_42:.*]] = fir.shape %[[VAL_41]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_43:.*]] = fir.array_load %[[VAL_44:.*]](%[[VAL_42]]) : (!fir.heap<!fir.array<4xi64>>, !fir.shape<1>) -> !fir.array<4xi64>
! CHECK:           %[[VAL_45:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_46:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_47:.*]] = arith.subi %[[VAL_3]], %[[VAL_45]] : index
! CHECK:           %[[VAL_48:.*]] = fir.do_loop %[[VAL_49:.*]] = %[[VAL_46]] to %[[VAL_47]] step %[[VAL_45]] unordered iter_args(%[[VAL_50:.*]] = %[[VAL_6]]) -> (!fir.array<4xi64>) {
! CHECK:             %[[VAL_51:.*]] = fir.array_fetch %[[VAL_43]], %[[VAL_49]] : (!fir.array<4xi64>, index) -> i64
! CHECK:             %[[VAL_52:.*]] = fir.no_reassoc %[[VAL_51]] : i64
! CHECK:             %[[VAL_53:.*]] = fir.array_update %[[VAL_50]], %[[VAL_52]], %[[VAL_49]] : (!fir.array<4xi64>, i64, index) -> !fir.array<4xi64>
! CHECK:             fir.result %[[VAL_53]] : !fir.array<4xi64>
! CHECK:           }
! CHECK:           fir.array_merge_store %[[VAL_6]], %[[VAL_54:.*]] to %[[VAL_4]] : !fir.array<4xi64>, !fir.array<4xi64>, !fir.ref<!fir.array<4xi64>>
! CHECK:           fir.freemem %[[VAL_44]] : !fir.heap<!fir.array<4xi64>>
! CHECK:           %[[VAL_55:.*]] = fir.load %[[VAL_4]] : !fir.ref<!fir.array<4xi64>>
! CHECK:           return %[[VAL_55]] : !fir.array<4xi64>
! CHECK:         }
