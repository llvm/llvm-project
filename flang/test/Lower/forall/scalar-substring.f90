! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

subroutine s(ch)
  character(10) :: ch
  forall (i=1:4)
     ch(i:i) = ch(i+1:i+1)
  end forall
end subroutine s
! CHECK-LABEL:   func.func @_QPs(
! CHECK-SAME:                    %[[VAL_0:.*]]: !fir.boxchar<1> {fir.bindc_name = "ch"}) {
! CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:           %[[VAL_2:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_3:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,10>>
! CHECK:           %[[VAL_4:.*]] = arith.constant 10 : index
! CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_3]] typeparams %[[VAL_4]] dummy_scope %[[VAL_1]] arg {{[0-9]+}} {uniq_name = "_QFsEch"} : (!fir.ref<!fir.char<1,10>>, index, !fir.dscope) -> (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,10>>)
! CHECK:           %[[VAL_6:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_7:.*]] = arith.constant 4 : i32
! CHECK:           hlfir.forall lb {
! CHECK:             hlfir.yield %[[VAL_6]] : i32
! CHECK:           } ub {
! CHECK:             hlfir.yield %[[VAL_7]] : i32
! CHECK:           }  (%[[VAL_8:.*]]: i32) {
! CHECK:             %[[VAL_9:.*]] = hlfir.forall_index "i" %[[VAL_8]] : (i32) -> !fir.ref<i32>
! CHECK:             hlfir.region_assign {
! CHECK:               %[[VAL_10:.*]] = fir.load %[[VAL_9]] : !fir.ref<i32>
! CHECK:               %[[VAL_11:.*]] = arith.constant 1 : i32
! CHECK:               %[[VAL_12:.*]] = arith.addi %[[VAL_10]], %[[VAL_11]] overflow<nsw> : i32
! CHECK:               %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i32) -> i64
! CHECK:               %[[VAL_14:.*]] = fir.load %[[VAL_9]] : !fir.ref<i32>
! CHECK:               %[[VAL_15:.*]] = arith.constant 1 : i32
! CHECK:               %[[VAL_16:.*]] = arith.addi %[[VAL_14]], %[[VAL_15]] overflow<nsw> : i32
! CHECK:               %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i32) -> i64
! CHECK:               %[[VAL_18:.*]] = fir.convert %[[VAL_13]] : (i64) -> index
! CHECK:               %[[VAL_19:.*]] = fir.convert %[[VAL_17]] : (i64) -> index
! CHECK:               %[[VAL_20:.*]] = arith.constant 1 : index
! CHECK:               %[[VAL_21:.*]] = arith.subi %[[VAL_19]], %[[VAL_18]] : index
! CHECK:               %[[VAL_22:.*]] = arith.addi %[[VAL_21]], %[[VAL_20]] : index
! CHECK:               %[[VAL_23:.*]] = arith.constant 0 : index
! CHECK:               %[[VAL_24:.*]] = arith.cmpi sgt, %[[VAL_22]], %[[VAL_23]] : index
! CHECK:               %[[VAL_25:.*]] = arith.select %[[VAL_24]], %[[VAL_22]], %[[VAL_23]] : index
! CHECK:               %[[VAL_26:.*]] = hlfir.designate %[[VAL_5]]#0  substr %[[VAL_18]], %[[VAL_19]]  typeparams %[[VAL_25]] : (!fir.ref<!fir.char<1,10>>, index, index, index) -> !fir.boxchar<1>
! CHECK:               hlfir.yield %[[VAL_26]] : !fir.boxchar<1>
! CHECK:             } to {
! CHECK:               %[[VAL_27:.*]] = fir.load %[[VAL_9]] : !fir.ref<i32>
! CHECK:               %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (i32) -> i64
! CHECK:               %[[VAL_29:.*]] = fir.load %[[VAL_9]] : !fir.ref<i32>
! CHECK:               %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (i32) -> i64
! CHECK:               %[[VAL_31:.*]] = fir.convert %[[VAL_28]] : (i64) -> index
! CHECK:               %[[VAL_32:.*]] = fir.convert %[[VAL_30]] : (i64) -> index
! CHECK:               %[[VAL_33:.*]] = arith.constant 1 : index
! CHECK:               %[[VAL_34:.*]] = arith.subi %[[VAL_32]], %[[VAL_31]] : index
! CHECK:               %[[VAL_35:.*]] = arith.addi %[[VAL_34]], %[[VAL_33]] : index
! CHECK:               %[[VAL_36:.*]] = arith.constant 0 : index
! CHECK:               %[[VAL_37:.*]] = arith.cmpi sgt, %[[VAL_35]], %[[VAL_36]] : index
! CHECK:               %[[VAL_38:.*]] = arith.select %[[VAL_37]], %[[VAL_35]], %[[VAL_36]] : index
! CHECK:               %[[VAL_39:.*]] = hlfir.designate %[[VAL_5]]#0  substr %[[VAL_31]], %[[VAL_32]]  typeparams %[[VAL_38]] : (!fir.ref<!fir.char<1,10>>, index, index, index) -> !fir.boxchar<1>
! CHECK:               hlfir.yield %[[VAL_39]] : !fir.boxchar<1>
! CHECK:             }
! CHECK:           }
! CHECK:           return
! CHECK:         }
