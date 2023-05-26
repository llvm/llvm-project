! RUN: bbc -emit-fir %s -o - | FileCheck %s

subroutine s(ch)
  character(10) :: ch
  forall (i=1:4)
     ch(i:i) = ch(i+1:i+1)
  end forall
end subroutine s

! CHECK-LABEL:   func.func @_QPs(
! CHECK-SAME:                    %[[VAL_0:.*]]: !fir.boxchar<1> {fir.bindc_name = "ch"}) {
! CHECK:           %[[VAL_1:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:           %[[VAL_2:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_3:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,10>>
! CHECK:           %[[VAL_4:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> index
! CHECK:           %[[VAL_6:.*]] = arith.constant 4 : i32
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
! CHECK:           %[[VAL_8:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_9:.*]] = %[[VAL_5]] to %[[VAL_7]] step %[[VAL_8]] unordered {
! CHECK:             %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (index) -> i32
! CHECK:             fir.store %[[VAL_10]] to %[[VAL_1]] : !fir.ref<i32>
! CHECK:             %[[VAL_11:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:             %[[VAL_12:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_13:.*]] = arith.addi %[[VAL_11]], %[[VAL_12]] : i32
! CHECK:             %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i32) -> i64
! CHECK:             %[[VAL_15:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:             %[[VAL_16:.*]] = arith.constant 1 : i32
! CHECK:             %[[VAL_17:.*]] = arith.addi %[[VAL_15]], %[[VAL_16]] : i32
! CHECK:             %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i32) -> i64
! CHECK:             %[[VAL_19:.*]] = fir.convert %[[VAL_14]] : (i64) -> index
! CHECK:             %[[VAL_20:.*]] = fir.convert %[[VAL_18]] : (i64) -> index
! CHECK:             %[[VAL_21:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_22:.*]] = arith.subi %[[VAL_19]], %[[VAL_21]] : index
! CHECK:             %[[VAL_23:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<!fir.array<10x!fir.char<1>>>
! CHECK:             %[[VAL_24:.*]] = fir.coordinate_of %[[VAL_23]], %[[VAL_22]] : (!fir.ref<!fir.array<10x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:             %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.char<1,?>>
! CHECK:             %[[VAL_26:.*]] = arith.subi %[[VAL_20]], %[[VAL_19]] : index
! CHECK:             %[[VAL_27:.*]] = arith.addi %[[VAL_26]], %[[VAL_21]] : index
! CHECK:             %[[VAL_28:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_29:.*]] = arith.cmpi slt, %[[VAL_27]], %[[VAL_28]] : index
! CHECK:             %[[VAL_30:.*]] = arith.select %[[VAL_29]], %[[VAL_28]], %[[VAL_27]] : index
! CHECK:             %[[VAL_31:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:             %[[VAL_32:.*]] = fir.convert %[[VAL_31]] : (i32) -> i64
! CHECK:             %[[VAL_33:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:             %[[VAL_34:.*]] = fir.convert %[[VAL_33]] : (i32) -> i64
! CHECK:             %[[VAL_35:.*]] = fir.convert %[[VAL_32]] : (i64) -> index
! CHECK:             %[[VAL_36:.*]] = fir.convert %[[VAL_34]] : (i64) -> index
! CHECK:             %[[VAL_37:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_38:.*]] = arith.subi %[[VAL_35]], %[[VAL_37]] : index
! CHECK:             %[[VAL_39:.*]] = fir.convert %[[VAL_3]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<!fir.array<10x!fir.char<1>>>
! CHECK:             %[[VAL_40:.*]] = fir.coordinate_of %[[VAL_39]], %[[VAL_38]] : (!fir.ref<!fir.array<10x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:             %[[VAL_41:.*]] = fir.convert %[[VAL_40]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.char<1,?>>
! CHECK:             %[[VAL_42:.*]] = arith.subi %[[VAL_36]], %[[VAL_35]] : index
! CHECK:             %[[VAL_43:.*]] = arith.addi %[[VAL_42]], %[[VAL_37]] : index
! CHECK:             %[[VAL_44:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_45:.*]] = arith.cmpi slt, %[[VAL_43]], %[[VAL_44]] : index
! CHECK:             %[[VAL_46:.*]] = arith.select %[[VAL_45]], %[[VAL_44]], %[[VAL_43]] : index
! CHECK:             %[[VAL_47:.*]] = arith.cmpi slt, %[[VAL_46]], %[[VAL_30]] : index
! CHECK:             %[[VAL_48:.*]] = arith.select %[[VAL_47]], %[[VAL_46]], %[[VAL_30]] : index
! CHECK:             %[[VAL_49:.*]] = arith.constant 1 : i64
! CHECK:             %[[VAL_50:.*]] = fir.convert %[[VAL_48]] : (index) -> i64
! CHECK:             %[[VAL_51:.*]] = arith.muli %[[VAL_49]], %[[VAL_50]] : i64
! CHECK:             %[[VAL_52:.*]] = arith.constant false
! CHECK:             %[[VAL_53:.*]] = fir.convert %[[VAL_41]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:             %[[VAL_54:.*]] = fir.convert %[[VAL_25]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:             fir.call @llvm.memmove.p0.p0.i64(%[[VAL_53]], %[[VAL_54]], %[[VAL_51]], %[[VAL_52]]) fastmath<contract> : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! CHECK:             %[[VAL_55:.*]] = arith.constant 1 : index
! CHECK:             %[[VAL_56:.*]] = arith.subi %[[VAL_46]], %[[VAL_55]] : index
! CHECK:             %[[VAL_57:.*]] = arith.constant 32 : i8
! CHECK:             %[[VAL_58:.*]] = fir.undefined !fir.char<1>
! CHECK:             %[[VAL_59:.*]] = fir.insert_value %[[VAL_58]], %[[VAL_57]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
! CHECK:             %[[VAL_60:.*]] = arith.constant 1 : index
! CHECK:             fir.do_loop %[[VAL_61:.*]] = %[[VAL_48]] to %[[VAL_56]] step %[[VAL_60]] {
! CHECK:               %[[VAL_62:.*]] = fir.convert %[[VAL_41]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
! CHECK:               %[[VAL_63:.*]] = fir.coordinate_of %[[VAL_62]], %[[VAL_61]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:               fir.store %[[VAL_59]] to %[[VAL_63]] : !fir.ref<!fir.char<1>>
! CHECK:             }
! CHECK:           }
! CHECK:           return
! CHECK:         }
