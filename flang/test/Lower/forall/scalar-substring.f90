! RUN: bbc -emit-fir %s -o - | FileCheck %s

subroutine s(ch)
  character(10) :: ch
  forall (i=1:4)
     ch(i:i) = ch(i+1:i+1)
  end forall
end subroutine s

! CHECK-LABEL: func @_QPs(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.boxchar<1> {fir.bindc_name = "ch"}) {
! CHECK:         %[[VAL_1:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_2:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:         %[[VAL_3:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (i32) -> index
! CHECK:         %[[VAL_5:.*]] = arith.constant 4 : i32
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> index
! CHECK:         %[[VAL_7:.*]] = arith.constant 1 : index
! CHECK:         fir.do_loop %[[VAL_8:.*]] = %[[VAL_4]] to %[[VAL_6]] step %[[VAL_7]] unordered {
! CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (index) -> i32
! CHECK:           fir.store %[[VAL_9]] to %[[VAL_1]] : !fir.ref<i32>
! CHECK:           %[[VAL_10:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:           %[[VAL_11:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_12:.*]] = arith.addi %[[VAL_10]], %[[VAL_11]] : i32
! CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i32) -> i64
! CHECK:           %[[VAL_14:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:           %[[VAL_15:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_16:.*]] = arith.addi %[[VAL_14]], %[[VAL_15]] : i32
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i32) -> i64
! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_13]] : (i64) -> index
! CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_17]] : (i64) -> index
! CHECK:           %[[VAL_20:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_21:.*]] = arith.subi %[[VAL_18]], %[[VAL_20]] : index
! CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
! CHECK:           %[[VAL_23:.*]] = fir.coordinate_of %[[VAL_22]], %[[VAL_21]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_23]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.char<1,?>>
! CHECK:           %[[VAL_25:.*]] = arith.subi %[[VAL_19]], %[[VAL_18]] : index
! CHECK:           %[[VAL_26:.*]] = arith.addi %[[VAL_25]], %[[VAL_20]] : index
! CHECK:           %[[VAL_27:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_28:.*]] = arith.cmpi slt, %[[VAL_26]], %[[VAL_27]] : index
! CHECK:           %[[VAL_29:.*]] = arith.select %[[VAL_28]], %[[VAL_27]], %[[VAL_26]] : index
! CHECK:           %[[VAL_30:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:           %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (i32) -> i64
! CHECK:           %[[VAL_32:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:           %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (i32) -> i64
! CHECK:           %[[VAL_34:.*]] = fir.convert %[[VAL_31]] : (i64) -> index
! CHECK:           %[[VAL_35:.*]] = fir.convert %[[VAL_33]] : (i64) -> index
! CHECK:           %[[VAL_36:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_37:.*]] = arith.subi %[[VAL_34]], %[[VAL_36]] : index
! CHECK:           %[[VAL_38:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
! CHECK:           %[[VAL_39:.*]] = fir.coordinate_of %[[VAL_38]], %[[VAL_37]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:           %[[VAL_40:.*]] = fir.convert %[[VAL_39]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.char<1,?>>
! CHECK:           %[[VAL_41:.*]] = arith.subi %[[VAL_35]], %[[VAL_34]] : index
! CHECK:           %[[VAL_42:.*]] = arith.addi %[[VAL_41]], %[[VAL_36]] : index
! CHECK:           %[[VAL_43:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_44:.*]] = arith.cmpi slt, %[[VAL_42]], %[[VAL_43]] : index
! CHECK:           %[[VAL_45:.*]] = arith.select %[[VAL_44]], %[[VAL_43]], %[[VAL_42]] : index
! CHECK:           %[[VAL_46:.*]] = arith.cmpi slt, %[[VAL_45]], %[[VAL_29]] : index
! CHECK:           %[[VAL_47:.*]] = arith.select %[[VAL_46]], %[[VAL_45]], %[[VAL_29]] : index
! CHECK:           %[[VAL_48:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_49:.*]] = fir.convert %[[VAL_47]] : (index) -> i64
! CHECK:           %[[VAL_50:.*]] = arith.muli %[[VAL_48]], %[[VAL_49]] : i64
! CHECK:           %[[VAL_51:.*]] = arith.constant false
! CHECK:           %[[VAL_52:.*]] = fir.convert %[[VAL_40]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_53:.*]] = fir.convert %[[VAL_24]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:           fir.call @llvm.memmove.p0.p0.i64(%[[VAL_52]], %[[VAL_53]], %[[VAL_50]], %[[VAL_51]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! CHECK:           %[[VAL_54:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_55:.*]] = arith.subi %[[VAL_45]], %[[VAL_54]] : index
! CHECK:           %[[VAL_56:.*]] = arith.constant 32 : i8
! CHECK:           %[[VAL_57:.*]] = fir.undefined !fir.char<1>
! CHECK:           %[[VAL_58:.*]] = fir.insert_value %[[VAL_57]], %[[VAL_56]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
! CHECK:           %[[VAL_59:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_60:.*]] = %[[VAL_47]] to %[[VAL_55]] step %[[VAL_59]] {
! CHECK:             %[[VAL_61:.*]] = fir.convert %[[VAL_40]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
! CHECK:             %[[VAL_62:.*]] = fir.coordinate_of %[[VAL_61]], %[[VAL_60]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:             fir.store %[[VAL_58]] to %[[VAL_62]] : !fir.ref<!fir.char<1>>
! CHECK:           }
! CHECK:         }
! CHECK:         return
! CHECK:       }

