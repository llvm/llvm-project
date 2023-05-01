! Test lowering of elemental calls with character argument
! without the VALUE attribute.
! RUN: bbc -o - %s | FileCheck %s

module char_elem

interface
elemental integer function elem(c)
  character(*), intent(in) :: c
end function

elemental integer function elem2(c, j)
  character(*), intent(in) :: c
  integer, intent(in) :: j
end function

end interface

contains

! CHECK-LABEL: func @_QMchar_elemPfoo1(
! CHECK-SAME: %[[VAL_15:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[VAL_4:.*]]: !fir.boxchar<1>{{.*}}) {
subroutine foo1(i, c)
  integer :: i(10)
  character(*) :: c(10)
! CHECK-DAG:   %[[VAL_0:.*]] = arith.constant 10 : index
! CHECK-DAG:   %[[VAL_1:.*]] = arith.constant 0 : index
! CHECK-DAG:   %[[VAL_2:.*]] = arith.constant 1 : index
! CHECK:   %[[VAL_3:.*]]:2 = fir.unboxchar %[[VAL_4]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:   %[[VAL_5:.*]] = fir.convert %[[VAL_3]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<10x!fir.char<1,?>>>
! CHECK:   %[[VAL_6:.*]] = fir.shape %[[VAL_0]] : (index) -> !fir.shape<1>
! CHECK:   br ^bb1(%[[VAL_1]], %[[VAL_0]] : index, index)
! CHECK: ^bb1(%[[VAL_7:.*]]: index, %[[VAL_8:.*]]: index):
! CHECK:   %[[VAL_9:.*]] = arith.cmpi sgt, %[[VAL_8]], %[[VAL_1]] : index
! CHECK:   cond_br %[[VAL_9]], ^bb2, ^bb3
! CHECK: ^bb2:
! CHECK:   %[[VAL_10:.*]] = arith.addi %[[VAL_7]], %[[VAL_2]] : index
! CHECK:   %[[VAL_11:.*]] = fir.array_coor %[[VAL_5]](%[[VAL_6]]) %[[VAL_10]] typeparams %[[VAL_3]]#1 : (!fir.ref<!fir.array<10x!fir.char<1,?>>>, !fir.shape<1>, index, index) -> !fir.ref<!fir.char<1,?>>
! CHECK:   %[[VAL_12:.*]] = fir.emboxchar %[[VAL_11]], %[[VAL_3]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:   %[[VAL_13:.*]] = fir.call @_QPelem(%[[VAL_12]]) {{.*}}: (!fir.boxchar<1>) -> i32
! CHECK:   %[[VAL_14:.*]] = fir.array_coor %[[VAL_15]](%[[VAL_6]]) %[[VAL_10]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
! CHECK:   fir.store %[[VAL_13]] to %[[VAL_14]] : !fir.ref<i32>
! CHECK:   %[[VAL_16:.*]] = arith.subi %[[VAL_8]], %[[VAL_2]] : index
! CHECK:   br ^bb1(%[[VAL_10]], %[[VAL_16]] : index, index)
! CHECK: ^bb3:
! CHECK:   return
  i = elem(c)
end subroutine

! CHECK-LABEL: func @_QMchar_elemPfoo1b(
! CHECK-SAME: %[[VAL_33:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[VAL_21:.*]]: !fir.boxchar<1>{{.*}}) {
subroutine foo1b(i, c)
  integer :: i(10)
  character(10) :: c(10)
! CHECK-DAG:   %[[VAL_17:.*]] = arith.constant 10 : index
! CHECK-DAG:   %[[VAL_18:.*]] = arith.constant 0 : index
! CHECK-DAG:   %[[VAL_19:.*]] = arith.constant 1 : index
! CHECK:   %[[VAL_20:.*]]:2 = fir.unboxchar %[[VAL_21]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:   %[[VAL_22:.*]] = fir.convert %[[VAL_20]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<10x!fir.char<1,10>>>
! CHECK:   %[[VAL_23:.*]] = fir.shape %[[VAL_17]] : (index) -> !fir.shape<1>
! CHECK:   br ^bb1(%[[VAL_18]], %[[VAL_17]] : index, index)
! CHECK: ^bb1(%[[VAL_24:.*]]: index, %[[VAL_25:.*]]: index):
! CHECK:   %[[VAL_26:.*]] = arith.cmpi sgt, %[[VAL_25]], %[[VAL_18]] : index
! CHECK:   cond_br %[[VAL_26]], ^bb2, ^bb3
! CHECK: ^bb2:
! CHECK:   %[[VAL_27:.*]] = arith.addi %[[VAL_24]], %[[VAL_19]] : index
! CHECK:   %[[VAL_28:.*]] = fir.array_coor %[[VAL_22]](%[[VAL_23]]) %[[VAL_27]] : (!fir.ref<!fir.array<10x!fir.char<1,10>>>, !fir.shape<1>, index) -> !fir.ref<!fir.char<1,10>>
! CHECK:   %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<!fir.char<1,?>>
! CHECK:   %[[VAL_30:.*]] = fir.emboxchar %[[VAL_29]], %[[VAL_17]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:   %[[VAL_31:.*]] = fir.call @_QPelem(%[[VAL_30]]) {{.*}}: (!fir.boxchar<1>) -> i32
! CHECK:   %[[VAL_32:.*]] = fir.array_coor %[[VAL_33]](%[[VAL_23]]) %[[VAL_27]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
! CHECK:   fir.store %[[VAL_31]] to %[[VAL_32]] : !fir.ref<i32>
! CHECK:   %[[VAL_34:.*]] = arith.subi %[[VAL_25]], %[[VAL_19]] : index
! CHECK:   br ^bb1(%[[VAL_27]], %[[VAL_34]] : index, index)
! CHECK: ^bb3:
! CHECK:   return
  i = elem(c)
end subroutine

! CHECK-LABEL: func @_QMchar_elemPfoo2(
! CHECK-SAME: %[[VAL_50:[^:]+]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[VAL_47:[^:]+]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[VAL_39:.*]]: !fir.boxchar<1>{{.*}}) {
subroutine foo2(i, j, c)
! CHECK-DAG:   %[[VAL_35:.*]] = arith.constant 10 : index
! CHECK-DAG:   %[[VAL_36:.*]] = arith.constant 0 : index
! CHECK-DAG:   %[[VAL_37:.*]] = arith.constant 1 : index
! CHECK:   %[[VAL_38:.*]]:2 = fir.unboxchar %[[VAL_39]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:   %[[VAL_40:.*]] = fir.shape %[[VAL_35]] : (index) -> !fir.shape<1>
! CHECK:   br ^bb1(%[[VAL_36]], %[[VAL_35]] : index, index)
! CHECK: ^bb1(%[[VAL_41:.*]]: index, %[[VAL_42:.*]]: index):
! CHECK:   %[[VAL_43:.*]] = arith.cmpi sgt, %[[VAL_42]], %[[VAL_36]] : index
! CHECK:   cond_br %[[VAL_43]], ^bb2, ^bb3
! CHECK: ^bb2:
! CHECK:   %[[VAL_44:.*]] = fir.emboxchar %[[VAL_38]]#0, %[[VAL_38]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:   %[[VAL_45:.*]] = arith.addi %[[VAL_41]], %[[VAL_37]] : index
! CHECK:   %[[VAL_46:.*]] = fir.array_coor %[[VAL_47]](%[[VAL_40]]) %[[VAL_45]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
! CHECK:   %[[VAL_48:.*]] = fir.call @_QPelem2(%[[VAL_44]], %[[VAL_46]]) {{.*}}: (!fir.boxchar<1>, !fir.ref<i32>) -> i32
! CHECK:   %[[VAL_49:.*]] = fir.array_coor %[[VAL_50]](%[[VAL_40]]) %[[VAL_45]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
! CHECK:   fir.store %[[VAL_48]] to %[[VAL_49]] : !fir.ref<i32>
! CHECK:   %[[VAL_51:.*]] = arith.subi %[[VAL_42]], %[[VAL_37]] : index
! CHECK:   br ^bb1(%[[VAL_45]], %[[VAL_51]] : index, index)
! CHECK: ^bb3:
! CHECK:   return
  integer :: i(10), j(10)
  character(*) :: c
  i = elem2(c, j)
end subroutine

! CHECK-LABEL:   func.func @_QMchar_elemPfoo2b(
! CHECK-SAME:        %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[VAL_2:.*]]: !fir.boxchar<1>{{.*}}) {
subroutine foo2b(i, j, c)
  integer :: i(10), j(10)
  character(10) :: c
! CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 10 : index
! CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_6:.*]]:2 = fir.unboxchar %[[VAL_2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:           %[[VAL_7:.*]] = fir.convert %[[VAL_6]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,10>>
! CHECK:           %[[VAL_8:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:           cf.br ^bb1(%[[VAL_4]], %[[VAL_3]] : index, index)
! CHECK:         ^bb1(%[[VAL_9:.*]]: index, %[[VAL_10:.*]]: index):
! CHECK:           %[[VAL_11:.*]] = arith.cmpi sgt, %[[VAL_10]], %[[VAL_4]] : index
! CHECK:           cf.cond_br %[[VAL_11]], ^bb2, ^bb3
! CHECK:         ^bb2:
! CHECK:           %[[VAL_12:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<!fir.char<1,?>>
! CHECK:           %[[VAL_13:.*]] = fir.emboxchar %[[VAL_12]], %[[VAL_3]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:           %[[VAL_14:.*]] = arith.addi %[[VAL_9]], %[[VAL_5]] : index
! CHECK:           %[[VAL_15:.*]] = fir.array_coor %[[VAL_1]](%[[VAL_8]]) %[[VAL_14]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
! CHECK:           %[[VAL_16:.*]] = fir.call @_QPelem2(%[[VAL_13]], %[[VAL_15]]) fastmath<contract> : (!fir.boxchar<1>, !fir.ref<i32>) -> i32
! CHECK:           %[[VAL_17:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_8]]) %[[VAL_14]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
! CHECK:           fir.store %[[VAL_16]] to %[[VAL_17]] : !fir.ref<i32>
! CHECK:           %[[VAL_18:.*]] = arith.subi %[[VAL_10]], %[[VAL_5]] : index
! CHECK:           cf.br ^bb1(%[[VAL_14]], %[[VAL_18]] : index, index)
! CHECK:         ^bb3:
! CHECK:           return
! CHECK:         }
  i = elem2(c, j)
end subroutine

! CHECK-LABEL: func @_QMchar_elemPfoo3(
! CHECK-SAME: %[[VAL_88:[^:]+]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[VAL_79:[^:]+]]: !fir.ref<!fir.array<10xi32>>{{.*}})
subroutine foo3(i, j)
  integer :: i(10), j(10)
! CHECK-DAG:   %[[VAL_69:.*]] = arith.constant 10 : index
! CHECK-DAG:   %[[VAL_70:.*]] = arith.constant 0 : index
! CHECK-DAG:   %[[VAL_71:.*]] = arith.constant 1 : index
! CHECK-DAG:   %[[VAL_72:.*]] = fir.alloca !fir.char<1>
! CHECK:   %[[VAL_73:.*]] = fir.shape %[[VAL_69]] : (index) -> !fir.shape<1>
! CHECK:   br ^bb1(%[[VAL_70]], %[[VAL_69]] : index, index)
! CHECK: ^bb1(%[[VAL_74:.*]]: index, %[[VAL_75:.*]]: index):
! CHECK:   %[[VAL_76:.*]] = arith.cmpi sgt, %[[VAL_75]], %[[VAL_70]] : index
! CHECK:   cond_br %[[VAL_76]], ^bb2, ^bb3
! CHECK: ^bb2:
! CHECK:   %[[VAL_77:.*]] = arith.addi %[[VAL_74]], %[[VAL_71]] : index
! CHECK:   %[[VAL_78:.*]] = fir.array_coor %[[VAL_79]](%[[VAL_73]]) %[[VAL_77]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
! CHECK:   %[[VAL_80:.*]] = fir.load %[[VAL_78]] : !fir.ref<i32>
! CHECK:   %[[VAL_81:.*]] = fir.convert %[[VAL_80]] : (i32) -> i8
! CHECK:   %[[VAL_82:.*]] = fir.undefined !fir.char<1>
! CHECK:   %[[VAL_83:.*]] = fir.insert_value %[[VAL_82]], %[[VAL_81]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
! CHECK:   fir.store %[[VAL_83]] to %[[VAL_72]] : !fir.ref<!fir.char<1>>
! CHECK:   %[[VAL_84:.*]] = fir.convert %[[VAL_72]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<!fir.char<1,?>>
! CHECK:   %[[VAL_85:.*]] = fir.emboxchar %[[VAL_84]], %[[VAL_71]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:   %[[VAL_86:.*]] = fir.call @_QPelem(%[[VAL_85]]) {{.*}}: (!fir.boxchar<1>) -> i32
! CHECK:   %[[VAL_87:.*]] = fir.array_coor %[[VAL_88]](%[[VAL_73]]) %[[VAL_77]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
! CHECK:   fir.store %[[VAL_86]] to %[[VAL_87]] : !fir.ref<i32>
! CHECK:   %[[VAL_89:.*]] = arith.subi %[[VAL_75]], %[[VAL_71]] : index
! CHECK:   br ^bb1(%[[VAL_77]], %[[VAL_89]] : index, index)
! CHECK: ^bb3:
! CHECK:   return
  i = elem(char(j))
end subroutine

! CHECK-LABEL: func @_QMchar_elemPfoo4(
! CHECK-SAME: %[[VAL_106:[^:]+]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[VAL_103:[^:]+]]: !fir.ref<!fir.array<10xi32>>{{.*}})
subroutine foo4(i, j)
  integer :: i(10), j(10)
! CHECK-DAG:   %[[VAL_90:.*]] = arith.constant 5 : index
! CHECK-DAG:   %[[VAL_91:.*]] = arith.constant 10 : index
! CHECK-DAG:   %[[VAL_92:.*]] = arith.constant 0 : index
! CHECK-DAG:   %[[VAL_93:.*]] = arith.constant 1 : index
! CHECK:   %[[VAL_94:.*]] = fir.shape %[[VAL_91]] : (index) -> !fir.shape<1>
! CHECK:   %[[VAL_95:.*]] = fir.address_of(@{{.*}}) : !fir.ref<!fir.char<1,5>>
! CHECK:   br ^bb1(%[[VAL_92]], %[[VAL_91]] : index, index)
! CHECK: ^bb1(%[[VAL_96:.*]]: index, %[[VAL_97:.*]]: index):
! CHECK:   %[[VAL_98:.*]] = arith.cmpi sgt, %[[VAL_97]], %[[VAL_92]] : index
! CHECK:   cond_br %[[VAL_98]], ^bb2, ^bb3
! CHECK: ^bb2:
! CHECK:   %[[VAL_99:.*]] = fir.convert %[[VAL_95]] : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<!fir.char<1,?>>
! CHECK:   %[[VAL_100:.*]] = fir.emboxchar %[[VAL_99]], %[[VAL_90]] : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:   %[[VAL_101:.*]] = arith.addi %[[VAL_96]], %[[VAL_93]] : index
! CHECK:   %[[VAL_102:.*]] = fir.array_coor %[[VAL_103]](%[[VAL_94]]) %[[VAL_101]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
! CHECK:   %[[VAL_104:.*]] = fir.call @_QPelem2(%[[VAL_100]], %[[VAL_102]]) {{.*}}: (!fir.boxchar<1>, !fir.ref<i32>) -> i32
! CHECK:   %[[VAL_105:.*]] = fir.array_coor %[[VAL_106]](%[[VAL_94]]) %[[VAL_101]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
! CHECK:   fir.store %[[VAL_104]] to %[[VAL_105]] : !fir.ref<i32>
! CHECK:   %[[VAL_107:.*]] = arith.subi %[[VAL_97]], %[[VAL_93]] : index
! CHECK:   br ^bb1(%[[VAL_101]], %[[VAL_107]] : index, index)
! CHECK: ^bb3:
! CHECK:   return
  i = elem2("hello", j)
end subroutine

! Test character return for elemental functions.

! CHECK-LABEL: func @_QMchar_elemPelem_return_char(
! CHECK-SAME: %{{.*}}: !fir.ref<!fir.char<1,?>>{{.*}}, %{{.*}}: index{{.*}}, %{{.*}}: !fir.boxchar<1>{{.*}}) -> !fir.boxchar<1>
elemental function elem_return_char(c)
 character(*), intent(in) :: c
 character(len(c)) :: elem_return_char
 elem_return_char = "ab" // c
end function

! CHECK-LABEL: func @_QMchar_elemPfoo6(
! CHECK-SAME:         %[[VAL_0:.*]]: !fir.boxchar<1> {fir.bindc_name = "c"}) {
subroutine foo6(c)
  ! CHECK-DAG:     %[[VAL_1:.*]] = arith.constant 10 : index
  ! CHECK-DAG:     %[[VAL_2:.*]] = arith.constant 1 : index
  ! CHECK-DAG:     %[[VAL_3:.*]] = arith.constant 0 : index
  ! CHECK-DAG:     %[[VAL_4:.*]] = arith.constant false
  ! CHECK-DAG:     %[[VAL_5:.*]] = arith.constant 32 : i8
  ! CHECK:         %[[VAL_6:.*]]:2 = fir.unboxchar %[[VAL_0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<10x!fir.char<1,?>>>
  ! CHECK:         %[[VAL_8:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
  ! CHECK:         br ^bb1(%[[VAL_3]], %[[VAL_1]] : index, index)
  ! CHECK:       ^bb1(%[[VAL_9:.*]]: index, %[[VAL_10:.*]]: index):
  ! CHECK:         %[[VAL_11:.*]] = arith.cmpi sgt, %[[VAL_10]], %[[VAL_3]] : index
  ! CHECK:         cond_br %[[VAL_11]], ^bb2, ^bb6
  ! CHECK:       ^bb2:
  ! CHECK:         %[[VAL_12:.*]] = arith.addi %[[VAL_9]], %[[VAL_2]] : index
  ! CHECK:         %[[VAL_13:.*]] = fir.array_coor %[[VAL_7]](%[[VAL_8]]) %[[VAL_12]] typeparams %[[VAL_6]]#1 : (!fir.ref<!fir.array<10x!fir.char<1,?>>>, !fir.shape<1>, index, index) -> !fir.ref<!fir.char<1,?>>
  ! CHECK:         %[[VAL_14:.*]] = fir.emboxchar %[[VAL_13]], %[[VAL_6]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK:         %[[VAL_15:.*]] = fir.convert %[[VAL_6]]#1 : (index) -> i32
  ! CHECK:         %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i32) -> index
  ! CHECK:         %[[CMPI:.*]] = arith.cmpi sgt, %[[VAL_16]], %{{.*}} : index
  ! CHECK:         %[[SELECT:.*]] = arith.select %[[CMPI]], %[[VAL_16]], %{{.*}} : index
  ! CHECK:         %[[VAL_17:.*]] = fir.call @llvm.stacksave() {{.*}}: () -> !fir.ref<i8>
  ! CHECK:         %[[VAL_18:.*]] = fir.alloca !fir.char<1,?>(%[[SELECT]] : index) {bindc_name = ".result"}
  ! CHECK:         %[[VAL_19:.*]] = fir.call @_QMchar_elemPelem_return_char(%[[VAL_18]], %[[SELECT]], %[[VAL_14]]) {{.*}}: (!fir.ref<!fir.char<1,?>>, index, !fir.boxchar<1>) -> !fir.boxchar<1>
  ! CHECK:         %[[VAL_20:.*]] = arith.cmpi slt, %[[VAL_6]]#1, %[[SELECT]] : index
  ! CHECK:         %[[VAL_21:.*]] = arith.select %[[VAL_20]], %[[VAL_6]]#1, %[[SELECT]] : index
  ! CHECK:         %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (index) -> i64
  ! CHECK:         %[[VAL_23:.*]] = fir.convert %[[VAL_13]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK:         %[[VAL_24:.*]] = fir.convert %[[VAL_18]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
  ! CHECK:         fir.call @llvm.memmove.p0.p0.i64(%[[VAL_23]], %[[VAL_24]], %[[VAL_22]], %[[VAL_4]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK:         %[[VAL_25:.*]] = arith.subi %[[VAL_6]]#1, %[[VAL_2]] : index
  ! CHECK:         %[[VAL_26:.*]] = fir.undefined !fir.char<1>
  ! CHECK:         %[[VAL_27:.*]] = fir.insert_value %[[VAL_26]], %[[VAL_5]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
  ! CHECK:         %[[VAL_28:.*]] = arith.subi %[[VAL_25]], %[[VAL_21]] : index
  ! CHECK:         %[[VAL_29:.*]] = arith.addi %[[VAL_28]], %[[VAL_2]] : index
  ! CHECK:         br ^bb3(%[[VAL_21]], %[[VAL_29]] : index, index)
  ! CHECK:       ^bb3(%[[VAL_30:.*]]: index, %[[VAL_31:.*]]: index):
  ! CHECK:         %[[VAL_32:.*]] = arith.cmpi sgt, %[[VAL_31]], %[[VAL_3]] : index
  ! CHECK:         cond_br %[[VAL_32]], ^bb4, ^bb5
  ! CHECK:       ^bb4:
  ! CHECK:         %[[VAL_33:.*]] = fir.convert %[[VAL_13]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<?x!fir.char<1>>>
  ! CHECK:         %[[VAL_34:.*]] = fir.coordinate_of %[[VAL_33]], %[[VAL_30]] : (!fir.ref<!fir.array<?x!fir.char<1>>>, index) -> !fir.ref<!fir.char<1>>
  ! CHECK:         fir.store %[[VAL_27]] to %[[VAL_34]] : !fir.ref<!fir.char<1>>
  ! CHECK:         %[[VAL_35:.*]] = arith.addi %[[VAL_30]], %[[VAL_2]] : index
  ! CHECK:         %[[VAL_36:.*]] = arith.subi %[[VAL_31]], %[[VAL_2]] : index
  ! CHECK:         br ^bb3(%[[VAL_35]], %[[VAL_36]] : index, index)
  ! CHECK:       ^bb5:
  ! CHECK:         fir.call @llvm.stackrestore(%[[VAL_17]]) {{.*}}: (!fir.ref<i8>) -> ()
  ! CHECK:         %[[VAL_37:.*]] = arith.subi %[[VAL_10]], %[[VAL_2]] : index
  ! CHECK:         br ^bb1(%[[VAL_12]], %[[VAL_37]] : index, index)
  ! CHECK:       ^bb6:

  implicit none
  character(*) :: c(10)
  c = elem_return_char(c)
  ! CHECK: return
  ! CHECK: }
end subroutine

end module
