! RUN: bbc -emit-fir %s -o - | FileCheck %s

subroutine ac1(arr,n)
  integer :: arr(:), n
  forall (i=1:n:2)
     arr(i:i+2) = func((/i/))
  end forall
contains
   pure integer function func(a)
    integer, intent(in) :: a(:)
    func = a(1)
  end function func
end subroutine ac1

! CHECK-LABEL: func @_QPac1(
! CHECK-SAME:               %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "arr"},
! CHECK-SAME:               %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca index {bindc_name = ".buff.pos"}
! CHECK:         %[[VAL_3:.*]] = fir.alloca index {bindc_name = ".buff.size"}
! CHECK:         %[[VAL_4:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_5:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> index
! CHECK:         %[[VAL_7:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> index
! CHECK:         %[[VAL_9:.*]] = arith.constant 2 : i32
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
! CHECK:         %[[VAL_11:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
! CHECK:         %[[VAL_12:.*]] = fir.do_loop %[[VAL_13:.*]] = %[[VAL_6]] to %[[VAL_8]] step %[[VAL_10]] unordered iter_args(%[[VAL_14:.*]] = %[[VAL_11]]) -> (!fir.array<?xi32>) {
! CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_13]] : (index) -> i32
! CHECK:           fir.store %[[VAL_15]] to %[[VAL_4]] : !fir.ref<i32>
! CHECK:           %[[VAL_16:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_17:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i32) -> i64
! CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i64) -> index
! CHECK:           %[[VAL_20:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i64) -> index
! CHECK:           %[[VAL_22:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
! CHECK:           %[[VAL_23:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_24:.*]] = arith.addi %[[VAL_22]], %[[VAL_23]] : i32
! CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_24]] : (i32) -> i64
! CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (i64) -> index
! CHECK:           %[[VAL_27:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_28:.*]] = arith.subi %[[VAL_26]], %[[VAL_19]] : index
! CHECK:           %[[VAL_29:.*]] = arith.addi %[[VAL_28]], %[[VAL_21]] : index
! CHECK:           %[[VAL_30:.*]] = arith.divsi %[[VAL_29]], %[[VAL_21]] : index
! CHECK:           %[[VAL_31:.*]] = arith.cmpi sgt, %[[VAL_30]], %[[VAL_27]] : index
! CHECK:           %[[VAL_32:.*]] = arith.select %[[VAL_31]], %[[VAL_30]], %[[VAL_27]] : index
! CHECK:           %[[VAL_33:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_34:.*]] = arith.constant 0 : index
! CHECK:           fir.store %[[VAL_34]] to %[[VAL_2]] : !fir.ref<index>
! CHECK:           %[[VAL_35:.*]] = fir.allocmem !fir.array<1xi32>
! CHECK:           %[[VAL_36:.*]] = arith.constant 1 : index
! CHECK:           fir.store %[[VAL_36]] to %[[VAL_3]] : !fir.ref<index>
! CHECK:           %[[VAL_37:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
! CHECK:           %[[VAL_38:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_39:.*]] = fir.zero_bits !fir.ref<!fir.array<1xi32>>
! CHECK:           %[[VAL_40:.*]] = fir.coordinate_of %[[VAL_39]], %[[VAL_38]] : (!fir.ref<!fir.array<1xi32>>, index) -> !fir.ref<i32>
! CHECK:           %[[VAL_41:.*]] = fir.convert %[[VAL_40]] : (!fir.ref<i32>) -> index
! CHECK:           %[[VAL_42:.*]] = fir.load %[[VAL_2]] : !fir.ref<index>
! CHECK:           %[[VAL_43:.*]] = fir.load %[[VAL_3]] : !fir.ref<index>
! CHECK:           %[[VAL_44:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_45:.*]] = arith.addi %[[VAL_42]], %[[VAL_44]] : index
! CHECK:           %[[VAL_46:.*]] = arith.cmpi sle, %[[VAL_43]], %[[VAL_45]] : index
! CHECK:           %[[VAL_47:.*]] = fir.if %[[VAL_46]] -> (!fir.heap<!fir.array<1xi32>>) {
! CHECK:             %[[VAL_48:.*]] = arith.constant 2 : index
! CHECK:             %[[VAL_49:.*]] = arith.muli %[[VAL_45]], %[[VAL_48]] : index
! CHECK:             fir.store %[[VAL_49]] to %[[VAL_3]] : !fir.ref<index>
! CHECK:             %[[VAL_50:.*]] = arith.muli %[[VAL_49]], %[[VAL_41]] : index
! CHECK:             %[[VAL_51:.*]] = fir.convert %[[VAL_35]] : (!fir.heap<!fir.array<1xi32>>) -> !fir.ref<i8>
! CHECK:             %[[VAL_52:.*]] = fir.convert %[[VAL_50]] : (index) -> i64
! CHECK:             %[[VAL_53:.*]] = fir.call @realloc(%[[VAL_51]], %[[VAL_52]]) : (!fir.ref<i8>, i64) -> !fir.ref<i8>
! CHECK:             %[[VAL_54:.*]] = fir.convert %[[VAL_53]] : (!fir.ref<i8>) -> !fir.heap<!fir.array<1xi32>>
! CHECK:             fir.result %[[VAL_54]] : !fir.heap<!fir.array<1xi32>>
! CHECK:           } else {
! CHECK:             fir.result %[[VAL_35]] : !fir.heap<!fir.array<1xi32>>
! CHECK:           }
! CHECK:           %[[VAL_55:.*]] = fir.coordinate_of %[[VAL_56:.*]], %[[VAL_42]] : (!fir.heap<!fir.array<1xi32>>, index) -> !fir.ref<i32>
! CHECK:           fir.store %[[VAL_37]] to %[[VAL_55]] : !fir.ref<i32>
! CHECK:           fir.store %[[VAL_45]] to %[[VAL_2]] : !fir.ref<index>
! CHECK:           %[[VAL_57:.*]] = fir.load %[[VAL_2]] : !fir.ref<index>
! CHECK:           %[[VAL_58:.*]] = fir.shape %[[VAL_57]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_59:.*]] = fir.array_load %[[VAL_56]](%[[VAL_58]]) : (!fir.heap<!fir.array<1xi32>>, !fir.shape<1>) -> !fir.array<1xi32>
! CHECK:           %[[VAL_60:.*]] = fir.allocmem !fir.array<1xi32>
! CHECK:           %[[VAL_61:.*]] = fir.shape %[[VAL_33]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_62:.*]] = fir.array_load %[[VAL_60]](%[[VAL_61]]) : (!fir.heap<!fir.array<1xi32>>, !fir.shape<1>) -> !fir.array<1xi32>
! CHECK:           %[[VAL_63:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_64:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_65:.*]] = arith.subi %[[VAL_33]], %[[VAL_63]] : index
! CHECK:           %[[VAL_66:.*]] = fir.do_loop %[[VAL_67:.*]] = %[[VAL_64]] to %[[VAL_65]] step %[[VAL_63]] unordered iter_args(%[[VAL_68:.*]] = %[[VAL_62]]) -> (!fir.array<1xi32>) {
! CHECK:             %[[VAL_69:.*]] = fir.array_fetch %[[VAL_59]], %[[VAL_67]] : (!fir.array<1xi32>, index) -> i32
! CHECK:             %[[VAL_70:.*]] = fir.array_update %[[VAL_68]], %[[VAL_69]], %[[VAL_67]] : (!fir.array<1xi32>, i32, index) -> !fir.array<1xi32>
! CHECK:             fir.result %[[VAL_70]] : !fir.array<1xi32>
! CHECK:           }
! CHECK:           fir.array_merge_store %[[VAL_62]], %[[VAL_71:.*]] to %[[VAL_60]] : !fir.array<1xi32>, !fir.array<1xi32>, !fir.heap<!fir.array<1xi32>>
! CHECK:           %[[VAL_72:.*]] = fir.shape %[[VAL_33]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_73:.*]] = fir.embox %[[VAL_60]](%[[VAL_72]]) : (!fir.heap<!fir.array<1xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<1xi32>>
! CHECK:           %[[VAL_74:.*]] = fir.convert %[[VAL_73]] : (!fir.box<!fir.array<1xi32>>) -> !fir.box<!fir.array<?xi32>>
! CHECK:           %[[VAL_75:.*]] = fir.call @_QFac1Pfunc(%[[VAL_74]]) : (!fir.box<!fir.array<?xi32>>) -> i32
! CHECK:           %[[VAL_76:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_77:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_78:.*]] = arith.subi %[[VAL_32]], %[[VAL_76]] : index
! CHECK:           %[[VAL_79:.*]] = fir.do_loop %[[VAL_80:.*]] = %[[VAL_77]] to %[[VAL_78]] step %[[VAL_76]] unordered iter_args(%[[VAL_81:.*]] = %[[VAL_14]]) -> (!fir.array<?xi32>) {
! CHECK:             %[[VAL_82:.*]] = arith.subi %[[VAL_19]], %[[VAL_16]] : index
! CHECK:             %[[VAL_83:.*]] = arith.muli %[[VAL_80]], %[[VAL_21]] : index
! CHECK:             %[[VAL_84:.*]] = arith.addi %[[VAL_82]], %[[VAL_83]] : index
! CHECK:             %[[VAL_85:.*]] = fir.array_update %[[VAL_81]], %[[VAL_75]], %[[VAL_84]] : (!fir.array<?xi32>, i32, index) -> !fir.array<?xi32>
! CHECK:             fir.result %[[VAL_85]] : !fir.array<?xi32>
! CHECK:           }
! CHECK:           fir.freemem %[[VAL_60]] : !fir.heap<!fir.array<1xi32>>
! CHECK:           fir.freemem %[[VAL_56]] : !fir.heap<!fir.array<1xi32>>
! CHECK:           fir.result %[[VAL_86:.*]] : !fir.array<?xi32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_11]], %[[VAL_87:.*]] to %[[VAL_0]] : !fir.array<?xi32>, !fir.array<?xi32>, !fir.box<!fir.array<?xi32>>
! CHECK:         return
! CHECK:       }

! CHECK-LABEL: func @_QFac1Pfunc(
! CHECK-SAME:                    %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}) -> i32 {
! CHECK:         %[[VAL_1:.*]] = fir.alloca i32 {bindc_name = "func", uniq_name = "_QFfuncEfunc"}
! CHECK:         %[[VAL_2:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_3:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_4:.*]] = arith.subi %[[VAL_2]], %[[VAL_3]] : i64
! CHECK:         %[[VAL_5:.*]] = fir.coordinate_of %[[VAL_0]], %[[VAL_4]] : (!fir.box<!fir.array<?xi32>>, i64) -> !fir.ref<i32>
! CHECK:         %[[VAL_6:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
! CHECK:         fir.store %[[VAL_6]] to %[[VAL_1]] : !fir.ref<i32>
! CHECK:         %[[VAL_7:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:         return %[[VAL_7]] : i32
! CHECK:       }

subroutine ac2(arr,n)
  integer :: arr(:), n
  forall (i=1:n:2)
     arr(i:i+2) = func((/i/))
  end forall
contains
  pure function func(a)
    integer :: func(3)
    integer, intent(in) :: a(:)
    func = a(1:3)
  end function func
end subroutine ac2

! CHECK-LABEL: func @_QPac2(
! CHECK-SAME:               %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "arr"},
! CHECK-SAME:               %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.array<3xi32> {bindc_name = ".result"}
! CHECK:         %[[VAL_3:.*]] = fir.alloca index {bindc_name = ".buff.pos"}
! CHECK:         %[[VAL_4:.*]] = fir.alloca index {bindc_name = ".buff.size"}
! CHECK:         %[[VAL_5:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_6:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
! CHECK:         %[[VAL_8:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i32) -> index
! CHECK:         %[[VAL_10:.*]] = arith.constant 2 : i32
! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i32) -> index
! CHECK:         %[[VAL_12:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
! CHECK:         %[[VAL_13:.*]] = fir.do_loop %[[VAL_14:.*]] = %[[VAL_7]] to %[[VAL_9]] step %[[VAL_11]] unordered iter_args(%[[VAL_15:.*]] = %[[VAL_12]]) -> (!fir.array<?xi32>) {
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_14]] : (index) -> i32
! CHECK:           fir.store %[[VAL_16]] to %[[VAL_5]] : !fir.ref<i32>
! CHECK:           %[[VAL_17:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_18:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
! CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i32) -> i64
! CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i64) -> index
! CHECK:           %[[VAL_21:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (i64) -> index
! CHECK:           %[[VAL_23:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
! CHECK:           %[[VAL_24:.*]] = arith.constant 2 : i32
! CHECK:           %[[VAL_25:.*]] = arith.addi %[[VAL_23]], %[[VAL_24]] : i32
! CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (i32) -> i64
! CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (i64) -> index
! CHECK:           %[[VAL_28:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_29:.*]] = arith.subi %[[VAL_27]], %[[VAL_20]] : index
! CHECK:           %[[VAL_30:.*]] = arith.addi %[[VAL_29]], %[[VAL_22]] : index
! CHECK:           %[[VAL_31:.*]] = arith.divsi %[[VAL_30]], %[[VAL_22]] : index
! CHECK:           %[[VAL_32:.*]] = arith.cmpi sgt, %[[VAL_31]], %[[VAL_28]] : index
! CHECK:           %[[VAL_33:.*]] = arith.select %[[VAL_32]], %[[VAL_31]], %[[VAL_28]] : index
! CHECK:           %[[VAL_34:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_35:.*]] = arith.constant 0 : index
! CHECK:           fir.store %[[VAL_35]] to %[[VAL_3]] : !fir.ref<index>
! CHECK:           %[[VAL_36:.*]] = fir.allocmem !fir.array<1xi32>
! CHECK:           %[[VAL_37:.*]] = arith.constant 1 : index
! CHECK:           fir.store %[[VAL_37]] to %[[VAL_4]] : !fir.ref<index>
! CHECK:           %[[VAL_38:.*]] = fir.load %[[VAL_5]] : !fir.ref<i32>
! CHECK:           %[[VAL_39:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_40:.*]] = fir.zero_bits !fir.ref<!fir.array<1xi32>>
! CHECK:           %[[VAL_41:.*]] = fir.coordinate_of %[[VAL_40]], %[[VAL_39]] : (!fir.ref<!fir.array<1xi32>>, index) -> !fir.ref<i32>
! CHECK:           %[[VAL_42:.*]] = fir.convert %[[VAL_41]] : (!fir.ref<i32>) -> index
! CHECK:           %[[VAL_43:.*]] = fir.load %[[VAL_3]] : !fir.ref<index>
! CHECK:           %[[VAL_44:.*]] = fir.load %[[VAL_4]] : !fir.ref<index>
! CHECK:           %[[VAL_45:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_46:.*]] = arith.addi %[[VAL_43]], %[[VAL_45]] : index
! CHECK:           %[[VAL_47:.*]] = arith.cmpi sle, %[[VAL_44]], %[[VAL_46]] : index
! CHECK:           %[[VAL_48:.*]] = fir.if %[[VAL_47]] -> (!fir.heap<!fir.array<1xi32>>) {
! CHECK:             %[[VAL_49:.*]] = arith.constant 2 : index
! CHECK:             %[[VAL_50:.*]] = arith.muli %[[VAL_46]], %[[VAL_49]] : index
! CHECK:             fir.store %[[VAL_50]] to %[[VAL_4]] : !fir.ref<index>
! CHECK:             %[[VAL_51:.*]] = arith.muli %[[VAL_50]], %[[VAL_42]] : index
! CHECK:             %[[VAL_52:.*]] = fir.convert %[[VAL_36]] : (!fir.heap<!fir.array<1xi32>>) -> !fir.ref<i8>
! CHECK:             %[[VAL_53:.*]] = fir.convert %[[VAL_51]] : (index) -> i64
! CHECK:             %[[VAL_54:.*]] = fir.call @realloc(%[[VAL_52]], %[[VAL_53]]) : (!fir.ref<i8>, i64) -> !fir.ref<i8>
! CHECK:             %[[VAL_55:.*]] = fir.convert %[[VAL_54]] : (!fir.ref<i8>) -> !fir.heap<!fir.array<1xi32>>
! CHECK:             fir.result %[[VAL_55]] : !fir.heap<!fir.array<1xi32>>
! CHECK:           } else {
! CHECK:             fir.result %[[VAL_36]] : !fir.heap<!fir.array<1xi32>>
! CHECK:           }
! CHECK:           %[[VAL_56:.*]] = fir.coordinate_of %[[VAL_57:.*]], %[[VAL_43]] : (!fir.heap<!fir.array<1xi32>>, index) -> !fir.ref<i32>
! CHECK:           fir.store %[[VAL_38]] to %[[VAL_56]] : !fir.ref<i32>
! CHECK:           fir.store %[[VAL_46]] to %[[VAL_3]] : !fir.ref<index>
! CHECK:           %[[VAL_58:.*]] = fir.load %[[VAL_3]] : !fir.ref<index>
! CHECK:           %[[VAL_59:.*]] = fir.shape %[[VAL_58]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_60:.*]] = fir.array_load %[[VAL_57]](%[[VAL_59]]) : (!fir.heap<!fir.array<1xi32>>, !fir.shape<1>) -> !fir.array<1xi32>
! CHECK:           %[[VAL_61:.*]] = fir.allocmem !fir.array<1xi32>
! CHECK:           %[[VAL_62:.*]] = fir.shape %[[VAL_34]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_63:.*]] = fir.array_load %[[VAL_61]](%[[VAL_62]]) : (!fir.heap<!fir.array<1xi32>>, !fir.shape<1>) -> !fir.array<1xi32>
! CHECK:           %[[VAL_64:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_65:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_66:.*]] = arith.subi %[[VAL_34]], %[[VAL_64]] : index
! CHECK:           %[[VAL_67:.*]] = fir.do_loop %[[VAL_68:.*]] = %[[VAL_65]] to %[[VAL_66]] step %[[VAL_64]] unordered iter_args(%[[VAL_69:.*]] = %[[VAL_63]]) -> (!fir.array<1xi32>) {
! CHECK:             %[[VAL_70:.*]] = fir.array_fetch %[[VAL_60]], %[[VAL_68]] : (!fir.array<1xi32>, index) -> i32
! CHECK:             %[[VAL_71:.*]] = fir.array_update %[[VAL_69]], %[[VAL_70]], %[[VAL_68]] : (!fir.array<1xi32>, i32, index) -> !fir.array<1xi32>
! CHECK:             fir.result %[[VAL_71]] : !fir.array<1xi32>
! CHECK:           }
! CHECK:           fir.array_merge_store %[[VAL_63]], %[[VAL_72:.*]] to %[[VAL_61]] : !fir.array<1xi32>, !fir.array<1xi32>, !fir.heap<!fir.array<1xi32>>
! CHECK:           %[[VAL_73:.*]] = fir.shape %[[VAL_34]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_74:.*]] = fir.embox %[[VAL_61]](%[[VAL_73]]) : (!fir.heap<!fir.array<1xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<1xi32>>
! CHECK:           %[[VAL_75:.*]] = arith.constant 3 : i64
! CHECK:           %[[VAL_76:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_77:.*]] = arith.subi %[[VAL_75]], %[[VAL_76]] : i64
! CHECK:           %[[VAL_78:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_79:.*]] = arith.addi %[[VAL_77]], %[[VAL_78]] : i64
! CHECK:           %[[VAL_80:.*]] = fir.convert %[[VAL_79]] : (i64) -> index
! CHECK:           %[[C0:.*]] = arith.constant 0 : index
! CHECK:           %[[CMPI:.*]] = arith.cmpi sgt, %[[VAL_80]], %[[C0]] : index
! CHECK:           %[[SELECT:.*]] = arith.select %[[CMPI]], %[[VAL_80]], %[[C0]] : index
! CHECK:           %[[VAL_81:.*]] = fir.call @llvm.stacksave() : () -> !fir.ref<i8>
! CHECK:           %[[VAL_82:.*]] = fir.shape %[[SELECT]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_83:.*]] = fir.convert %[[VAL_74]] : (!fir.box<!fir.array<1xi32>>) -> !fir.box<!fir.array<?xi32>>
! CHECK:           %[[VAL_84:.*]] = fir.call @_QFac2Pfunc(%[[VAL_83]]) : (!fir.box<!fir.array<?xi32>>) -> !fir.array<3xi32>
! CHECK:           fir.save_result %[[VAL_84]] to %[[VAL_2]](%[[VAL_82]]) : !fir.array<3xi32>, !fir.ref<!fir.array<3xi32>>, !fir.shape<1>
! CHECK:           %[[VAL_85:.*]] = fir.shape %[[SELECT]] : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_86:.*]] = fir.array_load %[[VAL_2]](%[[VAL_85]]) : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.array<3xi32>
! CHECK:           %[[VAL_87:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_88:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_89:.*]] = arith.subi %[[VAL_33]], %[[VAL_87]] : index
! CHECK:           %[[VAL_90:.*]] = fir.do_loop %[[VAL_91:.*]] = %[[VAL_88]] to %[[VAL_89]] step %[[VAL_87]] unordered iter_args(%[[VAL_92:.*]] = %[[VAL_15]]) -> (!fir.array<?xi32>) {
! CHECK:             %[[VAL_93:.*]] = fir.array_fetch %[[VAL_86]], %[[VAL_91]] : (!fir.array<3xi32>, index) -> i32
! CHECK:             %[[VAL_94:.*]] = arith.subi %[[VAL_20]], %[[VAL_17]] : index
! CHECK:             %[[VAL_95:.*]] = arith.muli %[[VAL_91]], %[[VAL_22]] : index
! CHECK:             %[[VAL_96:.*]] = arith.addi %[[VAL_94]], %[[VAL_95]] : index
! CHECK:             %[[VAL_97:.*]] = fir.array_update %[[VAL_92]], %[[VAL_93]], %[[VAL_96]] : (!fir.array<?xi32>, i32, index) -> !fir.array<?xi32>
! CHECK:             fir.result %[[VAL_97]] : !fir.array<?xi32>
! CHECK:           }
! CHECK:           fir.call @llvm.stackrestore(%[[VAL_81]]) : (!fir.ref<i8>) -> ()
! CHECK:           fir.freemem %[[VAL_61]] : !fir.heap<!fir.array<1xi32>>
! CHECK:           fir.freemem %[[VAL_57]] : !fir.heap<!fir.array<1xi32>>
! CHECK:           fir.result %[[VAL_98:.*]] : !fir.array<?xi32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_12]], %[[VAL_99:.*]] to %[[VAL_0]] : !fir.array<?xi32>, !fir.array<?xi32>, !fir.box<!fir.array<?xi32>>
! CHECK:         return
! CHECK:       }

! CHECK-LABEL: func @_QFac2Pfunc(
! CHECK-SAME:                    %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}) -> !fir.array<3xi32> {
! CHECK:         %[[VAL_1:.*]] = arith.constant 3 : index
! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.array<3xi32> {bindc_name = "func", uniq_name = "_QFfuncEfunc"}
! CHECK:         %[[VAL_3:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_4:.*]] = fir.array_load %[[VAL_2]](%[[VAL_3]]) : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> !fir.array<3xi32>
! CHECK:         %[[VAL_5:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i64) -> index
! CHECK:         %[[VAL_7:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i64) -> index
! CHECK:         %[[VAL_9:.*]] = arith.constant 3 : i64
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i64) -> index
! CHECK:         %[[VAL_11:.*]] = fir.slice %[[VAL_6]], %[[VAL_10]], %[[VAL_8]] : (index, index, index) -> !fir.slice<1>
! CHECK:         %[[VAL_12:.*]] = fir.array_load %[[VAL_0]] {{\[}}%[[VAL_11]]] : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>) -> !fir.array<?xi32>
! CHECK:         %[[VAL_13:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_14:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_15:.*]] = arith.subi %[[VAL_1]], %[[VAL_13]] : index
! CHECK:         %[[VAL_16:.*]] = fir.do_loop %[[VAL_17:.*]] = %[[VAL_14]] to %[[VAL_15]] step %[[VAL_13]] unordered iter_args(%[[VAL_18:.*]] = %[[VAL_4]]) -> (!fir.array<3xi32>) {
! CHECK:           %[[VAL_19:.*]] = fir.array_fetch %[[VAL_12]], %[[VAL_17]] : (!fir.array<?xi32>, index) -> i32
! CHECK:           %[[VAL_20:.*]] = fir.array_update %[[VAL_18]], %[[VAL_19]], %[[VAL_17]] : (!fir.array<3xi32>, i32, index) -> !fir.array<3xi32>
! CHECK:           fir.result %[[VAL_20]] : !fir.array<3xi32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_4]], %[[VAL_21:.*]] to %[[VAL_2]] : !fir.array<3xi32>, !fir.array<3xi32>, !fir.ref<!fir.array<3xi32>>
! CHECK:         %[[VAL_22:.*]] = fir.load %[[VAL_2]] : !fir.ref<!fir.array<3xi32>>
! CHECK:         return %[[VAL_22]] : !fir.array<3xi32>
! CHECK:       }
