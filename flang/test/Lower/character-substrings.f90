! Test character substring lowering
! RUN: bbc %s -o - -emit-fir | FileCheck %s

! Test substring lower where the parent is a scalar-char-literal-constant
! CHECK-LABEL: func @_QPscalar_substring_embox(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<i64>{{.*}}, %[[VAL_1:.*]]: !fir.ref<i64>{{.*}}) {
subroutine scalar_substring_embox(i, j)
  ! CHECK:         %[[VAL_2:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,18>>
  ! CHECK:         %[[VAL_3:.*]] = fir.load %[[VAL_0]] : !fir.ref<i64>
  ! CHECK:         %[[VAL_4:.*]] = fir.load %[[VAL_1]] : !fir.ref<i64>
  ! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_3]] : (i64) -> index
  ! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_4]] : (i64) -> index
  ! CHECK:         %[[VAL_7:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_8:.*]] = arith.subi %[[VAL_5]], %[[VAL_7]] : index
  ! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_2]] : (!fir.ref<!fir.char<1,18>>) -> !fir.ref<!fir.array<18x!fir.char<1>>>
  ! CHECK:         %[[VAL_10:.*]] = fir.coordinate_of %[[VAL_9]], %[[VAL_8]] : (!fir.ref<!fir.array<18x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
  ! CHECK:         %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK:         %[[VAL_12:.*]] = arith.subi %[[VAL_6]], %[[VAL_5]] : index
  ! CHECK:         %[[VAL_13:.*]] = arith.addi %[[VAL_12]], %[[VAL_7]] : index
  ! CHECK:         %[[VAL_14:.*]] = arith.constant 0 : index
  ! CHECK:         %[[VAL_15:.*]] = arith.cmpi slt, %[[VAL_13]], %[[VAL_14]] : index
  ! CHECK:         %[[VAL_16:.*]] = arith.select %[[VAL_15]], %[[VAL_14]], %[[VAL_13]] : index
  ! CHECK:         %[[VAL_17:.*]] = fir.emboxchar %[[VAL_11]], %[[VAL_16]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK:         fir.call @_QPbar(%[[VAL_17]]) {{.*}}: (!fir.boxchar<1>) -> ()
  integer(8) :: i, j
  call bar("abcHello World!dfg"(i:j))
  ! CHECK:         return
  ! CHECK:       }
end subroutine scalar_substring_embox

! CHECK-LABEL: func @_QParray_substring_embox(
! CHECK-SAME:        %[[VAL_0:.*]]: !fir.boxchar<1>{{.*}}) {
! CHECK:         %[[VAL_1:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:         %[[VAL_2:.*]] = fir.convert %[[VAL_1]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<4x!fir.char<1,7>>>
! CHECK:         %[[VAL_3:.*]] = arith.constant 4 : index
! CHECK:         %[[VAL_4:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_5:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i64) -> index
! CHECK:         %[[VAL_7:.*]] = arith.addi %[[VAL_4]], %[[VAL_3]] : index
! CHECK:         %[[VAL_8:.*]] = arith.subi %[[VAL_7]], %[[VAL_4]] : index
! CHECK:         %[[VAL_9:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_10:.*]] = arith.constant 5 : i64
! CHECK:         %[[VAL_11:.*]] = arith.constant 7 : i64
! CHECK:         %[[VAL_12:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_13:.*]] = arith.subi %[[VAL_10]], %[[VAL_12]] : i64
! CHECK:         %[[VAL_14:.*]] = arith.constant 0 : i64
! CHECK:         %[[VAL_15:.*]] = arith.subi %[[VAL_11]], %[[VAL_13]] : i64
! CHECK:         %[[VAL_16:.*]] = arith.cmpi sgt, %[[VAL_15]], %[[VAL_14]] : i64
! CHECK:         %[[VAL_17:.*]] = arith.select %[[VAL_16]], %[[VAL_15]], %[[VAL_14]] : i64
! CHECK:         %[[VAL_18:.*]] = fir.slice %[[VAL_4]], %[[VAL_8]], %[[VAL_6]] substr %[[VAL_13]], %[[VAL_17]] : (index, index, index, i64, i64) -> !fir.slice<1>
! CHECK:         %[[VAL_19:.*]] = fir.embox %[[VAL_2]](%[[VAL_9]]) {{\[}}%[[VAL_18]]] : (!fir.ref<!fir.array<4x!fir.char<1,7>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
! CHECK:         %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.box<!fir.array<?x!fir.char<1>>>
! CHECK:         fir.call @_QPs(%[[VAL_20]]) {{.*}}: (!fir.box<!fir.array<?x!fir.char<1>>>) -> ()
! CHECK:         return
! CHECK:       }

subroutine array_substring_embox(arr)
  interface
    subroutine s(a)
     character(1) :: a(:)
    end subroutine s
  end interface

  character(7) arr(4)

  call s(arr(:)(5:7))
end subroutine array_substring_embox

! CHECK-LABEL: func @_QPsubstring_assignment(
! CHECK-SAME:        %[[VAL_0:.*]]: !fir.boxchar<1>{{.*}}, %[[VAL_1:.*]]: !fir.boxchar<1>{{.*}}) {
subroutine substring_assignment(a,b)
  ! CHECK:         %[[VAL_2:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK:         %[[VAL_3:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK:         %[[VAL_4:.*]] = arith.constant 3 : i64
  ! CHECK:         %[[VAL_5:.*]] = arith.constant 4 : i64
  ! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_4]] : (i64) -> index
  ! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_5]] : (i64) -> index
  ! CHECK:         %[[VAL_8:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_9:.*]] = arith.subi %[[VAL_6]], %[[VAL_8]] : index
  ! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_3]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
  ! CHECK:         %[[VAL_11:.*]] = fir.coordinate_of %[[VAL_10]], %[[VAL_9]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
  ! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK:         %[[VAL_13:.*]] = arith.subi %[[VAL_7]], %[[VAL_6]] : index
  ! CHECK:         %[[VAL_14:.*]] = arith.addi %[[VAL_13]], %[[VAL_8]] : index
  ! CHECK:         %[[VAL_15:.*]] = arith.constant 0 : index
  ! CHECK:         %[[VAL_16:.*]] = arith.cmpi slt, %[[VAL_14]], %[[VAL_15]] : index
  ! CHECK:         %[[VAL_17:.*]] = arith.select %[[VAL_16]], %[[VAL_15]], %[[VAL_14]] : index
  ! CHECK:         %[[VAL_18:.*]] = arith.constant 1 : i64
  ! CHECK:         %[[VAL_19:.*]] = arith.constant 2 : i64
  ! CHECK:         %[[VAL_20:.*]] = fir.convert %[[VAL_18]] : (i64) -> index
  ! CHECK:         %[[VAL_21:.*]] = fir.convert %[[VAL_19]] : (i64) -> index
  ! CHECK:         %[[VAL_22:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_23:.*]] = arith.subi %[[VAL_20]], %[[VAL_22]] : index
  ! CHECK:         %[[VAL_24:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
  ! CHECK:         %[[VAL_25:.*]] = fir.coordinate_of %[[VAL_24]], %[[VAL_23]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
  ! CHECK:         %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.char<1,?>>
  ! CHECK:         %[[VAL_27:.*]] = arith.subi %[[VAL_21]], %[[VAL_20]] : index
  ! CHECK:         %[[VAL_28:.*]] = arith.addi %[[VAL_27]], %[[VAL_22]] : index
  ! CHECK:         %[[VAL_29:.*]] = arith.constant 0 : index
  ! CHECK:         %[[VAL_30:.*]] = arith.cmpi slt, %[[VAL_28]], %[[VAL_29]] : index
  ! CHECK:         %[[VAL_31:.*]] = arith.select %[[VAL_30]], %[[VAL_29]], %[[VAL_28]] : index
  ! CHECK:         %[[VAL_32:.*]] = arith.cmpi slt, %[[VAL_31]], %[[VAL_17]] : index
  ! CHECK:         %[[VAL_33:.*]] = arith.select %[[VAL_32]], %[[VAL_31]], %[[VAL_17]] : index
  ! CHECK:         %[[VAL_34:.*]] = arith.constant 1 : i64
  ! CHECK:         %[[VAL_35:.*]] = fir.convert %[[VAL_33]] : (index) -> i64
  ! CHECK:         %[[VAL_36:.*]] = arith.muli %[[VAL_34]], %[[VAL_35]] : i64
  ! CHECK:         %[[VAL_37:.*]] = arith.constant false
  ! CHECK:         %[[VAL_38:.*]] = fir.convert %[[VAL_26]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK:         %[[VAL_39:.*]] = fir.convert %[[VAL_12]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK:         fir.call @llvm.memmove.p0.p0.i64(%[[VAL_38]], %[[VAL_39]], %[[VAL_36]], %[[VAL_37]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK:         %[[VAL_40:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_41:.*]] = arith.subi %[[VAL_31]], %[[VAL_40]] : index
  ! CHECK:         %[[VAL_42:.*]] = arith.constant 32 : i8
  ! CHECK:         %[[VAL_43:.*]] = fir.undefined !fir.char<1>
  ! CHECK:         %[[VAL_44:.*]] = fir.insert_value %[[VAL_43]], %[[VAL_42]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
  ! CHECK:         %[[VAL_45:.*]] = arith.constant 1 : index
  ! CHECK:         fir.do_loop %[[VAL_46:.*]] = %[[VAL_33]] to %[[VAL_41]] step %[[VAL_45]] {
  ! CHECK:           %[[VAL_47:.*]] = fir.convert %[[VAL_26]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
  ! CHECK:           %[[VAL_48:.*]] = fir.coordinate_of %[[VAL_47]], %[[VAL_46]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
  ! CHECK:           fir.store %[[VAL_44]] to %[[VAL_48]] : !fir.ref<!fir.char<1>>
  ! CHECK:         }
  
  character(4) :: a, b
  a(1:2) = b(3:4)
  ! CHECK:         return
  ! CHECK:       }
end subroutine substring_assignment

! CHECK-LABEL: func @_QParray_substring_assignment(
! CHECK-SAME:        %[[VAL_0:.*]]: !fir.boxchar<1>{{.*}}) {
! CHECK:         %[[VAL_1:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:         %[[VAL_2:.*]] = fir.convert %[[VAL_1]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<6x!fir.char<1,5>>>
! CHECK:         %[[VAL_3:.*]] = arith.constant 6 : index
! CHECK:         %[[VAL_4:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_5:.*]] = arith.constant 1 : i64
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i64) -> index
! CHECK:         %[[VAL_7:.*]] = arith.addi %[[VAL_4]], %[[VAL_3]] : index
! CHECK:         %[[VAL_8:.*]] = arith.subi %[[VAL_7]], %[[VAL_4]] : index
! CHECK:         %[[VAL_9:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_10:.*]] = arith.subi %[[VAL_8]], %[[VAL_4]] : index
! CHECK:         %[[VAL_11:.*]] = arith.addi %[[VAL_10]], %[[VAL_6]] : index
! CHECK:         %[[VAL_12:.*]] = arith.divsi %[[VAL_11]], %[[VAL_6]] : index
! CHECK:         %[[VAL_13:.*]] = arith.cmpi sgt, %[[VAL_12]], %[[VAL_9]] : index
! CHECK:         %[[VAL_14:.*]] = arith.select %[[VAL_13]], %[[VAL_12]], %[[VAL_9]] : index
! CHECK:         %[[VAL_15:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_16:.*]] = fir.slice %[[VAL_4]], %[[VAL_8]], %[[VAL_6]] : (index, index, index) -> !fir.slice<1>
! CHECK:         %[[VAL_17:.*]] = fir.array_load %[[VAL_2]](%[[VAL_15]]) {{\[}}%[[VAL_16]]] : (!fir.ref<!fir.array<6x!fir.char<1,5>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<6x!fir.char<1,5>>
! CHECK:         %[[VAL_18:.*]] = fir.address_of(@_QQcl.424144) : !fir.ref<!fir.char<1,3>>
! CHECK:         %[[VAL_19:.*]] = arith.constant 3 : index
! CHECK:         %[[VAL_20:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_21:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_22:.*]] = arith.subi %[[VAL_14]], %[[VAL_20]] : index
! CHECK:         %[[VAL_23:.*]] = fir.do_loop %[[VAL_24:.*]] = %[[VAL_21]] to %[[VAL_22]] step %[[VAL_20]] unordered iter_args(%[[VAL_25:.*]] = %[[VAL_17]]) -> (!fir.array<6x!fir.char<1,5>>) {
! CHECK:           %[[VAL_26:.*]] = fir.array_access %[[VAL_25]], %[[VAL_24]] : (!fir.array<6x!fir.char<1,5>>, index) -> !fir.ref<!fir.char<1,5>>
! CHECK:           %[[VAL_27:.*]] = arith.constant 3 : i64
! CHECK:           %[[VAL_28:.*]] = arith.constant 5 : i64
! CHECK:           %[[VAL_29:.*]] = fir.convert %[[VAL_27]] : (i64) -> index
! CHECK:           %[[VAL_30:.*]] = fir.convert %[[VAL_28]] : (i64) -> index
! CHECK:           %[[VAL_31:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_32:.*]] = arith.subi %[[VAL_29]], %[[VAL_31]] : index
! CHECK:           %[[VAL_33:.*]] = fir.convert %[[VAL_26]] : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<!fir.array<5x!fir.char<1>>>
! CHECK:           %[[VAL_34:.*]] = fir.coordinate_of %[[VAL_33]], %[[VAL_32]] : (!fir.ref<!fir.array<5x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:           %[[VAL_35:.*]] = fir.convert %[[VAL_34]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.char<1,?>>
! CHECK:           %[[VAL_36:.*]] = arith.subi %[[VAL_30]], %[[VAL_29]] : index
! CHECK:           %[[VAL_37:.*]] = arith.addi %[[VAL_36]], %[[VAL_31]] : index
! CHECK:           %[[VAL_38:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_39:.*]] = arith.cmpi slt, %[[VAL_37]], %[[VAL_38]] : index
! CHECK:           %[[VAL_40:.*]] = arith.select %[[VAL_39]], %[[VAL_38]], %[[VAL_37]] : index
! CHECK:           %[[VAL_41:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_42:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_43:.*]] = fir.convert %[[VAL_40]] : (index) -> index
! CHECK:           %[[VAL_44:.*]] = arith.subi %[[VAL_43]], %[[VAL_42]] : index
! CHECK:           fir.do_loop %[[VAL_45:.*]] = %[[VAL_41]] to %[[VAL_44]] step %[[VAL_42]] {
! CHECK:             %[[VAL_46:.*]] = fir.convert %[[VAL_19]] : (index) -> index
! CHECK:             %[[VAL_47:.*]] = arith.cmpi slt, %[[VAL_45]], %[[VAL_46]] : index
! CHECK:             fir.if %[[VAL_47]] {
! CHECK:               %[[VAL_48:.*]] = fir.convert %[[VAL_18]] : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
! CHECK:               %[[VAL_49:.*]] = fir.coordinate_of %[[VAL_48]], %[[VAL_45]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:               %[[VAL_50:.*]] = fir.load %[[VAL_49]] : !fir.ref<!fir.char<1>>
! CHECK:               %[[VAL_51:.*]] = fir.convert %[[VAL_35]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
! CHECK:               %[[VAL_52:.*]] = fir.coordinate_of %[[VAL_51]], %[[VAL_45]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:               fir.store %[[VAL_50]] to %[[VAL_52]] : !fir.ref<!fir.char<1>>
! CHECK:             } else {
! CHECK:               %[[VAL_53:.*]] = fir.string_lit [32 : i8](1) : !fir.char<1>
! CHECK:               %[[VAL_54:.*]] = fir.convert %[[VAL_35]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
! CHECK:               %[[VAL_55:.*]] = fir.coordinate_of %[[VAL_54]], %[[VAL_45]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:               fir.store %[[VAL_53]] to %[[VAL_55]] : !fir.ref<!fir.char<1>>
! CHECK:             }
! CHECK:           }
! CHECK:           %[[VAL_56:.*]] = arith.cmpi slt, %[[VAL_40]], %[[VAL_19]] : index
! CHECK:           %[[VAL_57:.*]] = arith.select %[[VAL_56]], %[[VAL_40]], %[[VAL_19]] : index
! CHECK:           %[[VAL_58:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_59:.*]] = fir.convert %[[VAL_57]] : (index) -> i64
! CHECK:           %[[VAL_60:.*]] = arith.muli %[[VAL_58]], %[[VAL_59]] : i64
! CHECK:           %[[VAL_61:.*]] = arith.constant false
! CHECK:           %[[VAL_62:.*]] = fir.convert %[[VAL_35]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_63:.*]] = fir.convert %[[VAL_18]] : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<i8>
! CHECK:           fir.call @llvm.memmove.p0.p0.i64(%[[VAL_62]], %[[VAL_63]], %[[VAL_60]], %[[VAL_61]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! CHECK:           %[[VAL_64:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_65:.*]] = arith.subi %[[VAL_40]], %[[VAL_64]] : index
! CHECK:           %[[VAL_66:.*]] = arith.constant 32 : i8
! CHECK:           %[[VAL_67:.*]] = fir.undefined !fir.char<1>
! CHECK:           %[[VAL_68:.*]] = fir.insert_value %[[VAL_67]], %[[VAL_66]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
! CHECK:           %[[VAL_69:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_70:.*]] = %[[VAL_57]] to %[[VAL_65]] step %[[VAL_69]] {
! CHECK:             %[[VAL_71:.*]] = fir.convert %[[VAL_35]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
! CHECK:             %[[VAL_72:.*]] = fir.coordinate_of %[[VAL_71]], %[[VAL_70]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:             fir.store %[[VAL_68]] to %[[VAL_72]] : !fir.ref<!fir.char<1>>
! CHECK:           }
! CHECK:           %[[VAL_73:.*]] = fir.array_amend %[[VAL_25]], %[[VAL_26]] : (!fir.array<6x!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>) -> !fir.array<6x!fir.char<1,5>>
! CHECK:           fir.result %[[VAL_73]] : !fir.array<6x!fir.char<1,5>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_17]], %[[VAL_74:.*]] to %[[VAL_2]]{{\[}}%[[VAL_16]]] : !fir.array<6x!fir.char<1,5>>, !fir.array<6x!fir.char<1,5>>, !fir.ref<!fir.array<6x!fir.char<1,5>>>, !fir.slice<1>
! CHECK:         return
! CHECK:       }

subroutine array_substring_assignment(a)
  character(5) :: a(6)
  a(:)(3:5) = "BAD"
end subroutine array_substring_assignment

! CHECK-LABEL: func @_QParray_substring_assignment2(
! CHECK-SAME:        %[[VAL_0:.*]]: !fir.ref<!fir.array<8x!fir.type<_QFarray_substring_assignment2Tt{ch:!fir.char<1,7>}>>>{{.*}}) {
! CHECK:         %[[VAL_1:.*]] = arith.constant 8 : index
! CHECK:         %[[VAL_2:.*]] = fir.field_index ch, !fir.type<_QFarray_substring_assignment2Tt{ch:!fir.char<1,7>}>
! CHECK:         %[[VAL_3:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_4:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_5:.*]] = fir.slice %[[VAL_4]], %[[VAL_1]], %[[VAL_4]] path %[[VAL_2]] : (index, index, index, !fir.field) -> !fir.slice<1>
! CHECK:         %[[c0:.*]] = arith.constant 0 : index
! CHECK:         %[[sub:.*]] = arith.subi %[[VAL_1]], %[[VAL_4]] : index
! CHECK:         %[[add:.*]] = arith.addi %[[sub]], %[[VAL_4]] : index
! CHECK:         %[[div:.*]] = arith.divsi %4, %[[VAL_4]] : index
! CHECK:         %[[cmp:.*]] = arith.cmpi sgt, %[[div]], %[[c0]] : index
! CHECK:         %[[select:.*]] = arith.select %[[cmp]], %[[div]], %[[c0]] : index
! CHECK:         %[[VAL_6:.*]] = fir.array_load %[[VAL_0]](%[[VAL_3]]) {{\[}}%[[VAL_5]]] : (!fir.ref<!fir.array<8x!fir.type<_QFarray_substring_assignment2Tt{ch:!fir.char<1,7>}>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<8x!fir.char<1,7>>
! CHECK:         %[[VAL_7:.*]] = fir.address_of(@_QQcl.6E696365) : !fir.ref<!fir.char<1,4>>
! CHECK:         %[[VAL_8:.*]] = arith.constant 4 : index
! CHECK:         %[[VAL_9:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_10:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_11:.*]] = arith.subi %[[select]], %[[VAL_9]] : index
! CHECK:         %[[VAL_12:.*]] = fir.do_loop %[[VAL_13:.*]] = %[[VAL_10]] to %[[VAL_11]] step %[[VAL_9]] unordered iter_args(%[[VAL_14:.*]] = %[[VAL_6]]) -> (!fir.array<8x!fir.char<1,7>>) {
! CHECK:           %[[VAL_15:.*]] = fir.array_access %[[VAL_14]], %[[VAL_13]] : (!fir.array<8x!fir.char<1,7>>, index) -> !fir.ref<!fir.char<1,7>>
! CHECK:           %[[VAL_16:.*]] = arith.constant 4 : i64
! CHECK:           %[[VAL_17:.*]] = arith.constant 7 : i64
! CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_16]] : (i64) -> index
! CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_17]] : (i64) -> index
! CHECK:           %[[VAL_20:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_21:.*]] = arith.subi %[[VAL_18]], %[[VAL_20]] : index
! CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_15]] : (!fir.ref<!fir.char<1,7>>) -> !fir.ref<!fir.array<7x!fir.char<1>>>
! CHECK:           %[[VAL_23:.*]] = fir.coordinate_of %[[VAL_22]], %[[VAL_21]] : (!fir.ref<!fir.array<7x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_23]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.char<1,?>>
! CHECK:           %[[VAL_25:.*]] = arith.subi %[[VAL_19]], %[[VAL_18]] : index
! CHECK:           %[[VAL_26:.*]] = arith.addi %[[VAL_25]], %[[VAL_20]] : index
! CHECK:           %[[VAL_27:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_28:.*]] = arith.cmpi slt, %[[VAL_26]], %[[VAL_27]] : index
! CHECK:           %[[VAL_29:.*]] = arith.select %[[VAL_28]], %[[VAL_27]], %[[VAL_26]] : index
! CHECK:           %[[VAL_30:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_31:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_32:.*]] = fir.convert %[[VAL_29]] : (index) -> index
! CHECK:           %[[VAL_33:.*]] = arith.subi %[[VAL_32]], %[[VAL_31]] : index
! CHECK:           fir.do_loop %[[VAL_34:.*]] = %[[VAL_30]] to %[[VAL_33]] step %[[VAL_31]] {
! CHECK:             %[[VAL_35:.*]] = fir.convert %[[VAL_8]] : (index) -> index
! CHECK:             %[[VAL_36:.*]] = arith.cmpi slt, %[[VAL_34]], %[[VAL_35]] : index
! CHECK:             fir.if %[[VAL_36]] {
! CHECK:               %[[VAL_37:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.char<1,4>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
! CHECK:               %[[VAL_38:.*]] = fir.coordinate_of %[[VAL_37]], %[[VAL_34]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:               %[[VAL_39:.*]] = fir.load %[[VAL_38]] : !fir.ref<!fir.char<1>>
! CHECK:               %[[VAL_40:.*]] = fir.convert %[[VAL_24]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
! CHECK:               %[[VAL_41:.*]] = fir.coordinate_of %[[VAL_40]], %[[VAL_34]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:               fir.store %[[VAL_39]] to %[[VAL_41]] : !fir.ref<!fir.char<1>>
! CHECK:             } else {
! CHECK:               %[[VAL_42:.*]] = fir.string_lit [32 : i8](1) : !fir.char<1>
! CHECK:               %[[VAL_43:.*]] = fir.convert %[[VAL_24]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
! CHECK:               %[[VAL_44:.*]] = fir.coordinate_of %[[VAL_43]], %[[VAL_34]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:               fir.store %[[VAL_42]] to %[[VAL_44]] : !fir.ref<!fir.char<1>>
! CHECK:             }
! CHECK:           }
! CHECK:           %[[VAL_45:.*]] = arith.cmpi slt, %[[VAL_29]], %[[VAL_8]] : index
! CHECK:           %[[VAL_46:.*]] = arith.select %[[VAL_45]], %[[VAL_29]], %[[VAL_8]] : index
! CHECK:           %[[VAL_47:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_48:.*]] = fir.convert %[[VAL_46]] : (index) -> i64
! CHECK:           %[[VAL_49:.*]] = arith.muli %[[VAL_47]], %[[VAL_48]] : i64
! CHECK:           %[[VAL_50:.*]] = arith.constant false
! CHECK:           %[[VAL_51:.*]] = fir.convert %[[VAL_24]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_52:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.char<1,4>>) -> !fir.ref<i8>
! CHECK:           fir.call @llvm.memmove.p0.p0.i64(%[[VAL_51]], %[[VAL_52]], %[[VAL_49]], %[[VAL_50]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! CHECK:           %[[VAL_53:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_54:.*]] = arith.subi %[[VAL_29]], %[[VAL_53]] : index
! CHECK:           %[[VAL_55:.*]] = arith.constant 32 : i8
! CHECK:           %[[VAL_56:.*]] = fir.undefined !fir.char<1>
! CHECK:           %[[VAL_57:.*]] = fir.insert_value %[[VAL_56]], %[[VAL_55]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
! CHECK:           %[[VAL_58:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_59:.*]] = %[[VAL_46]] to %[[VAL_54]] step %[[VAL_58]] {
! CHECK:             %[[VAL_60:.*]] = fir.convert %[[VAL_24]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
! CHECK:             %[[VAL_61:.*]] = fir.coordinate_of %[[VAL_60]], %[[VAL_59]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:             fir.store %[[VAL_57]] to %[[VAL_61]] : !fir.ref<!fir.char<1>>
! CHECK:           }
! CHECK:           %[[VAL_62:.*]] = fir.array_amend %[[VAL_14]], %[[VAL_15]] : (!fir.array<8x!fir.char<1,7>>, !fir.ref<!fir.char<1,7>>) -> !fir.array<8x!fir.char<1,7>>
! CHECK:           fir.result %[[VAL_62]] : !fir.array<8x!fir.char<1,7>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_6]], %[[VAL_63:.*]] to %[[VAL_0]]{{\[}}%[[VAL_5]]] : !fir.array<8x!fir.char<1,7>>, !fir.array<8x!fir.char<1,7>>, !fir.ref<!fir.array<8x!fir.type<_QFarray_substring_assignment2Tt{ch:!fir.char<1,7>}>>>, !fir.slice<1>
! CHECK:         return
! CHECK:       }

subroutine array_substring_assignment2(a)
  type t
     character(7) :: ch
  end type t
  type(t) :: a(8)
  a%ch(4:7) = "nice"
end subroutine array_substring_assignment2

! CHECK-LABEL: func @_QParray_substring_assignment3(
! CHECK-SAME:     %[[VAL_0:.*]]: !fir.ref<!fir.array<8x!fir.type<_QFarray_substring_assignment3Tt{ch:!fir.char<1,7>}>>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<!fir.array<8x!fir.type<_QFarray_substring_assignment3Tt{ch:!fir.char<1,7>}>>>{{.*}}) {
! CHECK:         %[[VAL_2:.*]] = arith.constant 8 : index
! CHECK:         %[[VAL_3:.*]] = arith.constant 8 : index
! CHECK:         %[[VAL_4:.*]] = fir.field_index ch, !fir.type<_QFarray_substring_assignment3Tt{ch:!fir.char<1,7>}>
! CHECK:         %[[VAL_5:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_6:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_7:.*]] = fir.slice %[[VAL_6]], %[[VAL_2]], %[[VAL_6]] path %[[VAL_4]] : (index, index, index, !fir.field) -> !fir.slice<1>
! CHECK:         %[[c0:.*]] = arith.constant 0 : index
! CHECK:         %[[sub:.*]] = arith.subi %[[VAL_2]], %[[VAL_6]] : index
! CHECK:         %[[add:.*]] = arith.addi %[[sub]], %[[VAL_6]] : index
! CHECK:         %[[div:.*]] = arith.divsi %4, %[[VAL_6]] : index
! CHECK:         %[[cmp:.*]] = arith.cmpi sgt, %[[div]], %[[c0]] : index
! CHECK:         %[[select:.*]] = arith.select %[[cmp]], %[[div]], %[[c0]] : index
! CHECK:         %[[VAL_8:.*]] = fir.array_load %[[VAL_0]](%[[VAL_5]]) {{\[}}%[[VAL_7]]] : (!fir.ref<!fir.array<8x!fir.type<_QFarray_substring_assignment3Tt{ch:!fir.char<1,7>}>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<8x!fir.char<1,7>>
! CHECK:         %[[VAL_9:.*]] = fir.field_index ch, !fir.type<_QFarray_substring_assignment3Tt{ch:!fir.char<1,7>}>
! CHECK:         %[[VAL_10:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_11:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_12:.*]] = fir.slice %[[VAL_11]], %[[VAL_3]], %[[VAL_11]] path %[[VAL_9]] : (index, index, index, !fir.field) -> !fir.slice<1>
! CHECK:         %[[VAL_13:.*]] = fir.array_load %[[VAL_1]](%[[VAL_10]]) {{\[}}%[[VAL_12]]] : (!fir.ref<!fir.array<8x!fir.type<_QFarray_substring_assignment3Tt{ch:!fir.char<1,7>}>>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<8x!fir.char<1,7>>
! CHECK:         %[[VAL_14:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_15:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_16:.*]] = arith.subi %[[select]], %[[VAL_14]] : index
! CHECK:         %[[VAL_17:.*]] = fir.do_loop %[[VAL_18:.*]] = %[[VAL_15]] to %[[VAL_16]] step %[[VAL_14]] unordered iter_args(%[[VAL_19:.*]] = %[[VAL_8]]) -> (!fir.array<8x!fir.char<1,7>>) {
! CHECK:           %[[VAL_20:.*]] = fir.array_access %[[VAL_13]], %[[VAL_18]] : (!fir.array<8x!fir.char<1,7>>, index) -> !fir.ref<!fir.char<1,7>>
! CHECK:           %[[VAL_21:.*]] = arith.constant 2 : i64
! CHECK:           %[[VAL_22:.*]] = arith.constant 5 : i64
! CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_21]] : (i64) -> index
! CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_22]] : (i64) -> index
! CHECK:           %[[VAL_25:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_26:.*]] = arith.subi %[[VAL_23]], %[[VAL_25]] : index
! CHECK:           %[[VAL_27:.*]] = fir.convert %[[VAL_20]] : (!fir.ref<!fir.char<1,7>>) -> !fir.ref<!fir.array<7x!fir.char<1>>>
! CHECK:           %[[VAL_28:.*]] = fir.coordinate_of %[[VAL_27]], %[[VAL_26]] : (!fir.ref<!fir.array<7x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:           %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.char<1,?>>
! CHECK:           %[[VAL_30:.*]] = arith.subi %[[VAL_24]], %[[VAL_23]] : index
! CHECK:           %[[VAL_31:.*]] = arith.addi %[[VAL_30]], %[[VAL_25]] : index
! CHECK:           %[[VAL_32:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_33:.*]] = arith.cmpi slt, %[[VAL_31]], %[[VAL_32]] : index
! CHECK:           %[[VAL_34:.*]] = arith.select %[[VAL_33]], %[[VAL_32]], %[[VAL_31]] : index
! CHECK:           %[[VAL_35:.*]] = fir.array_access %[[VAL_19]], %[[VAL_18]] : (!fir.array<8x!fir.char<1,7>>, index) -> !fir.ref<!fir.char<1,7>>
! CHECK:           %[[VAL_36:.*]] = arith.constant 4 : i64
! CHECK:           %[[VAL_37:.*]] = arith.constant 7 : i64
! CHECK:           %[[VAL_38:.*]] = fir.convert %[[VAL_36]] : (i64) -> index
! CHECK:           %[[VAL_39:.*]] = fir.convert %[[VAL_37]] : (i64) -> index
! CHECK:           %[[VAL_40:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_41:.*]] = arith.subi %[[VAL_38]], %[[VAL_40]] : index
! CHECK:           %[[VAL_42:.*]] = fir.convert %[[VAL_35]] : (!fir.ref<!fir.char<1,7>>) -> !fir.ref<!fir.array<7x!fir.char<1>>>
! CHECK:           %[[VAL_43:.*]] = fir.coordinate_of %[[VAL_42]], %[[VAL_41]] : (!fir.ref<!fir.array<7x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:           %[[VAL_44:.*]] = fir.convert %[[VAL_43]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.char<1,?>>
! CHECK:           %[[VAL_45:.*]] = arith.subi %[[VAL_39]], %[[VAL_38]] : index
! CHECK:           %[[VAL_46:.*]] = arith.addi %[[VAL_45]], %[[VAL_40]] : index
! CHECK:           %[[VAL_47:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_48:.*]] = arith.cmpi slt, %[[VAL_46]], %[[VAL_47]] : index
! CHECK:           %[[VAL_49:.*]] = arith.select %[[VAL_48]], %[[VAL_47]], %[[VAL_46]] : index
! CHECK:           %[[VAL_50:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_51:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_52:.*]] = fir.convert %[[VAL_49]] : (index) -> index
! CHECK:           %[[VAL_53:.*]] = arith.subi %[[VAL_52]], %[[VAL_51]] : index
! CHECK:           fir.do_loop %[[VAL_54:.*]] = %[[VAL_50]] to %[[VAL_53]] step %[[VAL_51]] {
! CHECK:             %[[VAL_55:.*]] = fir.convert %[[VAL_34]] : (index) -> index
! CHECK:             %[[VAL_56:.*]] = arith.cmpi slt, %[[VAL_54]], %[[VAL_55]] : index
! CHECK:             fir.if %[[VAL_56]] {
! CHECK:               %[[VAL_57:.*]] = fir.convert %[[VAL_29]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
! CHECK:               %[[VAL_58:.*]] = fir.coordinate_of %[[VAL_57]], %[[VAL_54]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:               %[[VAL_59:.*]] = fir.load %[[VAL_58]] : !fir.ref<!fir.char<1>>
! CHECK:               %[[VAL_60:.*]] = fir.convert %[[VAL_44]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
! CHECK:               %[[VAL_61:.*]] = fir.coordinate_of %[[VAL_60]], %[[VAL_54]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:               fir.store %[[VAL_59]] to %[[VAL_61]] : !fir.ref<!fir.char<1>>
! CHECK:             } else {
! CHECK:               %[[VAL_62:.*]] = fir.string_lit [32 : i8](1) : !fir.char<1>
! CHECK:               %[[VAL_63:.*]] = fir.convert %[[VAL_44]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
! CHECK:               %[[VAL_64:.*]] = fir.coordinate_of %[[VAL_63]], %[[VAL_54]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:               fir.store %[[VAL_62]] to %[[VAL_64]] : !fir.ref<!fir.char<1>>
! CHECK:             }
! CHECK:           }
! CHECK:           %[[VAL_65:.*]] = arith.cmpi slt, %[[VAL_49]], %[[VAL_34]] : index
! CHECK:           %[[VAL_66:.*]] = arith.select %[[VAL_65]], %[[VAL_49]], %[[VAL_34]] : index
! CHECK:           %[[VAL_67:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_68:.*]] = fir.convert %[[VAL_66]] : (index) -> i64
! CHECK:           %[[VAL_69:.*]] = arith.muli %[[VAL_67]], %[[VAL_68]] : i64
! CHECK:           %[[VAL_70:.*]] = arith.constant false
! CHECK:           %[[VAL_71:.*]] = fir.convert %[[VAL_44]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:           %[[VAL_72:.*]] = fir.convert %[[VAL_29]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:           fir.call @llvm.memmove.p0.p0.i64(%[[VAL_71]], %[[VAL_72]], %[[VAL_69]], %[[VAL_70]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! CHECK:           %[[VAL_73:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_74:.*]] = arith.subi %[[VAL_49]], %[[VAL_73]] : index
! CHECK:           %[[VAL_75:.*]] = arith.constant 32 : i8
! CHECK:           %[[VAL_76:.*]] = fir.undefined !fir.char<1>
! CHECK:           %[[VAL_77:.*]] = fir.insert_value %[[VAL_76]], %[[VAL_75]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
! CHECK:           %[[VAL_78:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_79:.*]] = %[[VAL_66]] to %[[VAL_74]] step %[[VAL_78]] {
! CHECK:             %[[VAL_80:.*]] = fir.convert %[[VAL_44]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
! CHECK:             %[[VAL_81:.*]] = fir.coordinate_of %[[VAL_80]], %[[VAL_79]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
! CHECK:             fir.store %[[VAL_77]] to %[[VAL_81]] : !fir.ref<!fir.char<1>>
! CHECK:           }
! CHECK:           %[[VAL_82:.*]] = fir.array_amend %[[VAL_19]], %[[VAL_35]] : (!fir.array<8x!fir.char<1,7>>, !fir.ref<!fir.char<1,7>>) -> !fir.array<8x!fir.char<1,7>>
! CHECK:           fir.result %[[VAL_82]] : !fir.array<8x!fir.char<1,7>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_8]], %[[VAL_83:.*]] to %[[VAL_0]]{{\[}}%[[VAL_7]]] : !fir.array<8x!fir.char<1,7>>, !fir.array<8x!fir.char<1,7>>, !fir.ref<!fir.array<8x!fir.type<_QFarray_substring_assignment3Tt{ch:!fir.char<1,7>}>>>, !fir.slice<1>
! CHECK:         return
! CHECK:       }


subroutine array_substring_assignment3(a,b)
  type t
     character(7) :: ch
  end type t
  type(t) :: a(8), b(8)
  a%ch(4:7) = b%ch(2:5)
end subroutine array_substring_assignment3
