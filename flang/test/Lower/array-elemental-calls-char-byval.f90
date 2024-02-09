! Test lowering of elemental calls with character argument
! with the VALUE attribute.
! RUN: bbc -hlfir=false -o - %s | FileCheck %s


module char_elem_byval

interface
elemental integer function elem(c, j)
  character(*), value :: c
  integer, intent(in) :: j
end function
end interface

contains
! CHECK-LABEL: func @_QMchar_elem_byvalPfoo1(
! CHECK-SAME: %[[VAL_22:[^:]+]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[VAL_19:[^:]+]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[VAL_5:.*]]: !fir.boxchar<1>{{.*}}) {
subroutine foo1(i, j, c)
  integer :: i(10), j(10)
  character(*) :: c(10)
! CHECK-DAG:   %[[VAL_0:.*]] = arith.constant false
! CHECK-DAG:   %[[VAL_1:.*]] = arith.constant 10 : index
! CHECK-DAG:   %[[VAL_2:.*]] = arith.constant 0 : index
! CHECK-DAG:   %[[VAL_3:.*]] = arith.constant 1 : index
! CHECK:   %[[VAL_4:.*]]:2 = fir.unboxchar %[[VAL_5]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:   %[[VAL_6:.*]] = fir.convert %[[VAL_4]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<10x!fir.char<1,?>>>
! CHECK:   %[[VAL_7:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
! CHECK:   br ^bb1(%[[VAL_2]], %[[VAL_1]] : index, index)
! CHECK: ^bb1(%[[VAL_8:.*]]: index, %[[VAL_9:.*]]: index):
! CHECK:   %[[VAL_10:.*]] = arith.cmpi sgt, %[[VAL_9]], %[[VAL_2]] : index
! CHECK:   cond_br %[[VAL_10]], ^bb2, ^bb3
! CHECK: ^bb2:
! CHECK:   %[[VAL_11:.*]] = arith.addi %[[VAL_8]], %[[VAL_3]] : index
! CHECK:   %[[VAL_12:.*]] = fir.array_coor %[[VAL_6]](%[[VAL_7]]) %[[VAL_11]] typeparams %[[VAL_4]]#1 : (!fir.ref<!fir.array<10x!fir.char<1,?>>>, !fir.shape<1>, index, index) -> !fir.ref<!fir.char<1,?>>
! CHECK:   %[[VAL_13:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_4]]#1 : index) {bindc_name = ".chrtmp"}
! CHECK:   %[[VAL_14:.*]] = fir.convert %[[VAL_4]]#1 : (index) -> i64
! CHECK:   %[[VAL_15:.*]] = fir.convert %[[VAL_13]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:   %[[VAL_16:.*]] = fir.convert %[[VAL_12]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:   fir.call @llvm.memmove.p0.p0.i64(%[[VAL_15]], %[[VAL_16]], %[[VAL_14]], %[[VAL_0]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! CHECK:   %[[VAL_17:.*]] = fir.emboxchar %[[VAL_13]], %[[VAL_4]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:   %[[VAL_18:.*]] = fir.array_coor %[[VAL_19]](%[[VAL_7]]) %[[VAL_11]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
! CHECK:   %[[VAL_20:.*]] = fir.call @_QPelem(%[[VAL_17]], %[[VAL_18]]) {{.*}}: (!fir.boxchar<1>, !fir.ref<i32>) -> i32
! CHECK:   %[[VAL_21:.*]] = fir.array_coor %[[VAL_22]](%[[VAL_7]]) %[[VAL_11]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
! CHECK:   fir.store %[[VAL_20]] to %[[VAL_21]] : !fir.ref<i32>
! CHECK:   %[[VAL_23:.*]] = arith.subi %[[VAL_9]], %[[VAL_3]] : index
! CHECK:   br ^bb1(%[[VAL_11]], %[[VAL_23]] : index, index)
! CHECK: ^bb3:
! CHECK:   return
  i = elem(c, j)
end subroutine

! CHECK-LABEL: func @_QMchar_elem_byvalPfoo2(
! CHECK-SAME: %[[VAL_44:[^:]+]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[VAL_41:[^:]+]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[VAL_29:.*]]: !fir.boxchar<1>{{.*}}) {
subroutine foo2(i, j, c)
  integer :: i(10), j(10)
  character(*) :: c
! CHECK-DAG:   %[[VAL_24:.*]] = arith.constant false
! CHECK-DAG:   %[[VAL_25:.*]] = arith.constant 10 : index
! CHECK-DAG:   %[[VAL_26:.*]] = arith.constant 0 : index
! CHECK-DAG:   %[[VAL_27:.*]] = arith.constant 1 : index
! CHECK:   %[[VAL_28:.*]]:2 = fir.unboxchar %[[VAL_29]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:   %[[VAL_30:.*]] = fir.shape %[[VAL_25]] : (index) -> !fir.shape<1>
! CHECK:   %[[VAL_31:.*]] = fir.alloca !fir.char<1,?>(%[[VAL_28]]#1 : index) {bindc_name = ".chrtmp"}
! CHECK:   %[[VAL_32:.*]] = fir.convert %[[VAL_28]]#1 : (index) -> i64
! CHECK:   %[[VAL_33:.*]] = fir.convert %[[VAL_31]] : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:   %[[VAL_34:.*]] = fir.convert %[[VAL_28]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:   fir.call @llvm.memmove.p0.p0.i64(%[[VAL_33]], %[[VAL_34]], %[[VAL_32]], %[[VAL_24]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! CHECK:   br ^bb1(%[[VAL_26]], %[[VAL_25]] : index, index)
! CHECK: ^bb1(%[[VAL_35:.*]]: index, %[[VAL_36:.*]]: index):
! CHECK:   %[[VAL_37:.*]] = arith.cmpi sgt, %[[VAL_36]], %[[VAL_26]] : index
! CHECK:   cond_br %[[VAL_37]], ^bb2, ^bb3
! CHECK: ^bb2:
! CHECK:   %[[VAL_38:.*]] = fir.emboxchar %[[VAL_31]], %[[VAL_28]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
! CHECK:   %[[VAL_39:.*]] = arith.addi %[[VAL_35]], %[[VAL_27]] : index
! CHECK:   %[[VAL_40:.*]] = fir.array_coor %[[VAL_41]](%[[VAL_30]]) %[[VAL_39]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
! CHECK:   %[[VAL_42:.*]] = fir.call @_QPelem(%[[VAL_38]], %[[VAL_40]]) {{.*}}: (!fir.boxchar<1>, !fir.ref<i32>) -> i32
! CHECK:   %[[VAL_43:.*]] = fir.array_coor %[[VAL_44]](%[[VAL_30]]) %[[VAL_39]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
! CHECK:   fir.store %[[VAL_42]] to %[[VAL_43]] : !fir.ref<i32>
! CHECK:   %[[VAL_45:.*]] = arith.subi %[[VAL_36]], %[[VAL_27]] : index
! CHECK:   br ^bb1(%[[VAL_39]], %[[VAL_45]] : index, index)
! CHECK: ^bb3:
! CHECK:   return
  i = elem(c, j)
end subroutine

! CHECK-LABEL: func @_QMchar_elem_byvalPfoo3(
! CHECK-SAME: %[[VAL_65:[^:]+]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[VAL_55:[^:]+]]: !fir.ref<!fir.array<10xi32>>{{.*}}) {
subroutine foo3(i, j)
  integer :: i(10), j(10)
! CHECK-DAG:   %[[VAL_46:.*]] = arith.constant 10 : index
! CHECK-DAG:   %[[VAL_47:.*]] = arith.constant 0 : index
! CHECK-DAG:   %[[VAL_48:.*]] = arith.constant 1 : index
! CHECK:   %[[VAL_49:.*]] = fir.shape %[[VAL_46]] : (index) -> !fir.shape<1>
! CHECK:   br ^bb1(%[[VAL_47]], %[[VAL_46]] : index, index)
! CHECK: ^bb1(%[[VAL_50:.*]]: index, %[[VAL_51:.*]]: index):
! CHECK:   %[[VAL_52:.*]] = arith.cmpi sgt, %[[VAL_51]], %[[VAL_47]] : index
! CHECK:   cond_br %[[VAL_52]], ^bb2, ^bb3
! CHECK: ^bb2:
! CHECK:   %[[VAL_53:.*]] = arith.addi %[[VAL_50]], %[[VAL_48]] : index
! CHECK:   %[[VAL_54:.*]] = fir.array_coor %[[VAL_55]](%[[VAL_49]]) %[[VAL_53]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
! CHECK:   %[[VAL_56:.*]] = fir.load %[[VAL_54]] : !fir.ref<i32>
! CHECK:   %[[VAL_57:.*]] = fir.convert %[[VAL_56]] : (i32) -> i8
! CHECK:   %[[VAL_58:.*]] = fir.undefined !fir.char<1>
! CHECK:   %[[VAL_59:.*]] = fir.insert_value %[[VAL_58]], %[[VAL_57]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
! CHECK:   %[[VAL_60:.*]] = fir.alloca !fir.char<1> {bindc_name = ".chrtmp"}
! CHECK:   fir.store %[[VAL_59]] to %[[VAL_60]] : !fir.ref<!fir.char<1>>
! CHECK:   %[[VAL_62:.*]] = fir.emboxchar %[[VAL_60]], %[[VAL_48]] : (!fir.ref<!fir.char<1>>, index) -> !fir.boxchar<1>
! CHECK:   %[[VAL_63:.*]] = fir.call @_QPelem(%[[VAL_62]], %[[VAL_54]]) {{.*}}: (!fir.boxchar<1>, !fir.ref<i32>) -> i32
! CHECK:   %[[VAL_64:.*]] = fir.array_coor %[[VAL_65]](%[[VAL_49]]) %[[VAL_53]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
! CHECK:   fir.store %[[VAL_63]] to %[[VAL_64]] : !fir.ref<i32>
! CHECK:   %[[VAL_66:.*]] = arith.subi %[[VAL_51]], %[[VAL_48]] : index
! CHECK:   br ^bb1(%[[VAL_53]], %[[VAL_66]] : index, index)
! CHECK: ^bb3:
! CHECK:   return
  i = elem(char(j), j)
end subroutine

! CHECK-LABEL: func @_QMchar_elem_byvalPfoo4(
! CHECK-SAME: %[[VAL_93:[^:]+]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[VAL_74:[^:]+]]: !fir.ref<!fir.array<10xi32>>{{.*}}) {
subroutine foo4(i, j)
  integer :: i(10), j(10)
! CHECK-DAG:   %[[VAL_67:.*]] = arith.constant 0 : i64
! CHECK-DAG:   %[[VAL_68:.*]] = arith.constant false
! CHECK-DAG:   %[[VAL_69:.*]] = arith.constant 10 : index
! CHECK-DAG:   %[[VAL_70:.*]] = arith.constant 0 : index
! CHECK-DAG:   %[[VAL_71:.*]] = arith.constant 1 : index
! CHECK:   %[[VAL_66:.*]] = fir.alloca !fir.char<1> {adapt.valuebyref}
! CHECK:   %[[VAL_72:.*]] = fir.shape %[[VAL_69]] : (index) -> !fir.shape<1>
! CHECK:   %[[VAL_73:.*]] = fir.coordinate_of %[[VAL_74]], %[[VAL_67]] : (!fir.ref<!fir.array<10xi32>>, i64) -> !fir.ref<i32>
! CHECK:   %[[VAL_75:.*]] = fir.load %[[VAL_73]] : !fir.ref<i32>
! CHECK:   %[[VAL_76:.*]] = fir.convert %[[VAL_75]] : (i32) -> i8
! CHECK:   %[[VAL_77:.*]] = fir.undefined !fir.char<1>
! CHECK:   %[[VAL_78:.*]] = fir.insert_value %[[VAL_77]], %[[VAL_76]], [0 : index] : (!fir.char<1>, i8) -> !fir.char<1>
! CHECK:   fir.store %[[VAL_78]] to %[[VAL_66]] : !fir.ref<!fir.char<1>>
! CHECK:   %[[VAL_80:.*]] = fir.alloca !fir.char<1> {bindc_name = ".chrtmp"}
! CHECK:   %[[VAL_81:.*]] = fir.convert %[[VAL_71]] : (index) -> i64
! CHECK:   %[[VAL_82:.*]] = fir.convert %[[VAL_80]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<i8>
! CHECK:   %[[VAL_83:.*]] = fir.convert %[[VAL_66]] : (!fir.ref<!fir.char<1>>) -> !fir.ref<i8>
! CHECK:   fir.call @llvm.memmove.p0.p0.i64(%[[VAL_82]], %[[VAL_83]], %[[VAL_81]], %[[VAL_68]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! CHECK:   br ^bb1(%[[VAL_70]], %[[VAL_69]] : index, index)
! CHECK: ^bb1(%[[VAL_84:.*]]: index, %[[VAL_85:.*]]: index):
! CHECK:   %[[VAL_86:.*]] = arith.cmpi sgt, %[[VAL_85]], %[[VAL_70]] : index
! CHECK:   cond_br %[[VAL_86]], ^bb2, ^bb3
! CHECK: ^bb2:
! CHECK:   %[[VAL_88:.*]] = fir.emboxchar %[[VAL_80]], %[[VAL_71]] : (!fir.ref<!fir.char<1>>, index) -> !fir.boxchar<1>
! CHECK:   %[[VAL_89:.*]] = arith.addi %[[VAL_84]], %[[VAL_71]] : index
! CHECK:   %[[VAL_90:.*]] = fir.array_coor %[[VAL_74]](%[[VAL_72]]) %[[VAL_89]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
! CHECK:   %[[VAL_91:.*]] = fir.call @_QPelem(%[[VAL_88]], %[[VAL_90]]) {{.*}}: (!fir.boxchar<1>, !fir.ref<i32>) -> i32
! CHECK:   %[[VAL_92:.*]] = fir.array_coor %[[VAL_93]](%[[VAL_72]]) %[[VAL_89]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
! CHECK:   fir.store %[[VAL_91]] to %[[VAL_92]] : !fir.ref<i32>
! CHECK:   %[[VAL_94:.*]] = arith.subi %[[VAL_85]], %[[VAL_71]] : index
! CHECK:   br ^bb1(%[[VAL_89]], %[[VAL_94]] : index, index)
! CHECK: ^bb3:
! CHECK:   return
  i = elem(char(j(1)), j)
end subroutine

! Note: the copy of the constant is important because VALUE argument can be
! modified on the caller side.

! CHECK-LABEL: func @_QMchar_elem_byvalPfoo5(
! CHECK-SAME: %[[VAL_116:[^:]+]]: !fir.ref<!fir.array<10xi32>>{{.*}}, %[[VAL_113:[^:]+]]: !fir.ref<!fir.array<10xi32>>{{.*}}) {
subroutine foo5(i, j)
  integer :: i(10), j(10)
! CHECK-DAG:   %[[VAL_95:.*]] = arith.constant 5 : index
! CHECK-DAG:   %[[VAL_96:.*]] = arith.constant false
! CHECK-DAG:   %[[VAL_97:.*]] = arith.constant 10 : index
! CHECK-DAG:   %[[VAL_98:.*]] = arith.constant 0 : index
! CHECK-DAG:   %[[VAL_99:.*]] = arith.constant 1 : index
! CHECK:   %[[VAL_100:.*]] = fir.shape %[[VAL_97]] : (index) -> !fir.shape<1>
! CHECK:   %[[VAL_101:.*]] = fir.address_of(@{{.*}}) : !fir.ref<!fir.char<1,5>>
! CHECK:   %[[VAL_102:.*]] = fir.alloca !fir.char<1,5> {bindc_name = ".chrtmp"}
! CHECK:   %[[VAL_103:.*]] = fir.convert %[[VAL_95]] : (index) -> i64
! CHECK:   %[[VAL_104:.*]] = fir.convert %[[VAL_102]] : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<i8>
! CHECK:   %[[VAL_105:.*]] = fir.convert %[[VAL_101]] : (!fir.ref<!fir.char<1,5>>) -> !fir.ref<i8>
! CHECK:   fir.call @llvm.memmove.p0.p0.i64(%[[VAL_104]], %[[VAL_105]], %[[VAL_103]], %[[VAL_96]]) {{.*}}: (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! CHECK:   br ^bb1(%[[VAL_98]], %[[VAL_97]] : index, index)
! CHECK: ^bb1(%[[VAL_106:.*]]: index, %[[VAL_107:.*]]: index):
! CHECK:   %[[VAL_108:.*]] = arith.cmpi sgt, %[[VAL_107]], %[[VAL_98]] : index
! CHECK:   cond_br %[[VAL_108]], ^bb2, ^bb3
! CHECK: ^bb2:
! CHECK:   %[[VAL_110:.*]] = fir.emboxchar %[[VAL_102]], %[[VAL_95]] : (!fir.ref<!fir.char<1,5>>, index) -> !fir.boxchar<1>
! CHECK:   %[[VAL_111:.*]] = arith.addi %[[VAL_106]], %[[VAL_99]] : index
! CHECK:   %[[VAL_112:.*]] = fir.array_coor %[[VAL_113]](%[[VAL_100]]) %[[VAL_111]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
! CHECK:   %[[VAL_114:.*]] = fir.call @_QPelem(%[[VAL_110]], %[[VAL_112]]) {{.*}}: (!fir.boxchar<1>, !fir.ref<i32>) -> i32
! CHECK:   %[[VAL_115:.*]] = fir.array_coor %[[VAL_116]](%[[VAL_100]]) %[[VAL_111]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
! CHECK:   fir.store %[[VAL_114]] to %[[VAL_115]] : !fir.ref<i32>
! CHECK:   %[[VAL_117:.*]] = arith.subi %[[VAL_107]], %[[VAL_99]] : index
! CHECK:   br ^bb1(%[[VAL_111]], %[[VAL_117]] : index, index)
! CHECK: ^bb3:
! CHECK:   return
  i = elem("hello", j)
end subroutine

end module
