! Test derived type assignment lowering inside array expression
! RUN: bbc %s -o - | FileCheck %s

module array_derived_assign
  type simple_copy
    integer :: i
    character(10) :: c(20)
    real, pointer :: p(:)
  end type
  type deep_copy
    integer :: i
    real, allocatable :: a(:)
  end type
contains

! Simple copies are implemented inline component by component.
! CHECK-LABEL: func @_QMarray_derived_assignPtest_simple_copy(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.ref<!fir.array<10x!fir.type<_QMarray_derived_assignTsimple_copy{i:i32,c:!fir.array<20x!fir.char<1,10>>,p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<!fir.array<10x!fir.type<_QMarray_derived_assignTsimple_copy{i:i32,c:!fir.array<20x!fir.char<1,10>>,p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>>{{.*}}) {
subroutine test_simple_copy(t1, t2)
  type(simple_copy) :: t1(10), t2(10)
  ! CHECK-DAG:         %[[VAL_2:.*]] = arith.constant 20 : index
  ! CHECK-DAG:         %[[VAL_3:.*]] = arith.constant 10 : index
  ! CHECK-DAG:         %[[VAL_4:.*]] = arith.constant false
  ! CHECK-DAG:         %[[VAL_5:.*]] = arith.constant 0 : index
  ! CHECK-DAG:         %[[VAL_6:.*]] = arith.constant 1 : index
  ! CHECK-DAG:         %[[VAL_7:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
  ! CHECK:         br ^bb1(%[[VAL_5]], %[[VAL_3]] : index, index)
  ! CHECK:       ^bb1(%[[VAL_8:.*]]: index, %[[VAL_9:.*]]: index):
  ! CHECK:         %[[VAL_10:.*]] = arith.cmpi sgt, %[[VAL_9]], %[[VAL_5]] : index
  ! CHECK:         cond_br %[[VAL_10]], ^bb2, ^bb6
  ! CHECK:       ^bb2:
  ! CHECK:         %[[VAL_11:.*]] = arith.addi %[[VAL_8]], %[[VAL_6]] : index
  ! CHECK:         %[[VAL_12:.*]] = fir.array_coor %[[VAL_1]](%[[VAL_7]]) %[[VAL_11]] : (!fir.ref<!fir.array<10x!fir.type<_QMarray_derived_assignTsimple_copy{i:i32,c:!fir.array<20x!fir.char<1,10>>,p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QMarray_derived_assignTsimple_copy{i:i32,c:!fir.array<20x!fir.char<1,10>>,p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>
  ! CHECK:         %[[VAL_13:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_7]]) %[[VAL_11]] : (!fir.ref<!fir.array<10x!fir.type<_QMarray_derived_assignTsimple_copy{i:i32,c:!fir.array<20x!fir.char<1,10>>,p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QMarray_derived_assignTsimple_copy{i:i32,c:!fir.array<20x!fir.char<1,10>>,p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>
  ! CHECK:         %[[VAL_14:.*]] = fir.field_index i, !fir.type<_QMarray_derived_assignTsimple_copy{i:i32,c:!fir.array<20x!fir.char<1,10>>,p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>
  ! CHECK:         %[[VAL_15:.*]] = fir.coordinate_of %[[VAL_12]], %[[VAL_14]] : (!fir.ref<!fir.type<_QMarray_derived_assignTsimple_copy{i:i32,c:!fir.array<20x!fir.char<1,10>>,p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>, !fir.field) -> !fir.ref<i32>
  ! CHECK:         %[[VAL_16:.*]] = fir.coordinate_of %[[VAL_13]], %[[VAL_14]] : (!fir.ref<!fir.type<_QMarray_derived_assignTsimple_copy{i:i32,c:!fir.array<20x!fir.char<1,10>>,p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>, !fir.field) -> !fir.ref<i32>
  ! CHECK:         %[[VAL_17:.*]] = fir.load %[[VAL_15]] : !fir.ref<i32>
  ! CHECK:         fir.store %[[VAL_17]] to %[[VAL_16]] : !fir.ref<i32>
  ! CHECK:         %[[VAL_18:.*]] = fir.field_index c, !fir.type<_QMarray_derived_assignTsimple_copy{i:i32,c:!fir.array<20x!fir.char<1,10>>,p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>
  ! CHECK:         %[[VAL_19:.*]] = fir.coordinate_of %[[VAL_12]], %[[VAL_18]] : (!fir.ref<!fir.type<_QMarray_derived_assignTsimple_copy{i:i32,c:!fir.array<20x!fir.char<1,10>>,p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>, !fir.field) -> !fir.ref<!fir.array<20x!fir.char<1,10>>>
  ! CHECK:         %[[VAL_20:.*]] = fir.coordinate_of %[[VAL_13]], %[[VAL_18]] : (!fir.ref<!fir.type<_QMarray_derived_assignTsimple_copy{i:i32,c:!fir.array<20x!fir.char<1,10>>,p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>, !fir.field) -> !fir.ref<!fir.array<20x!fir.char<1,10>>>
  ! CHECK:         br ^bb3(%[[VAL_5]], %[[VAL_2]] : index, index)
  ! CHECK:       ^bb3(%[[VAL_21:.*]]: index, %[[VAL_22:.*]]: index):
  ! CHECK:         %[[VAL_23:.*]] = arith.cmpi sgt, %[[VAL_22]], %[[VAL_5]] : index
  ! CHECK:         cond_br %[[VAL_23]], ^bb4, ^bb5
  ! CHECK:       ^bb4:
  ! CHECK:         %[[VAL_24:.*]] = fir.coordinate_of %[[VAL_20]], %[[VAL_21]] : (!fir.ref<!fir.array<20x!fir.char<1,10>>>, index) -> !fir.ref<!fir.char<1,10>>
  ! CHECK:         %[[VAL_25:.*]] = fir.coordinate_of %[[VAL_19]], %[[VAL_21]] : (!fir.ref<!fir.array<20x!fir.char<1,10>>>, index) -> !fir.ref<!fir.char<1,10>>
  ! CHECK:         %[[VAL_26:.*]] = fir.convert %[[VAL_3]] : (index) -> i64
  ! CHECK:         %[[VAL_27:.*]] = fir.convert %[[VAL_24]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
  ! CHECK:         %[[VAL_28:.*]] = fir.convert %[[VAL_25]] : (!fir.ref<!fir.char<1,10>>) -> !fir.ref<i8>
  ! CHECK:         fir.call @llvm.memmove.p0.p0.i64(%[[VAL_27]], %[[VAL_28]], %[[VAL_26]], %[[VAL_4]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK:         %[[VAL_29:.*]] = arith.addi %[[VAL_21]], %[[VAL_6]] : index
  ! CHECK:         %[[VAL_30:.*]] = arith.subi %[[VAL_22]], %[[VAL_6]] : index
  ! CHECK:         br ^bb3(%[[VAL_29]], %[[VAL_30]] : index, index)
  ! CHECK:       ^bb5:
  ! CHECK:         %[[VAL_31:.*]] = fir.field_index p, !fir.type<_QMarray_derived_assignTsimple_copy{i:i32,c:!fir.array<20x!fir.char<1,10>>,p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>
  ! CHECK:         %[[VAL_32:.*]] = fir.coordinate_of %[[VAL_12]], %[[VAL_31]] : (!fir.ref<!fir.type<_QMarray_derived_assignTsimple_copy{i:i32,c:!fir.array<20x!fir.char<1,10>>,p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK:         %[[VAL_33:.*]] = fir.coordinate_of %[[VAL_13]], %[[VAL_31]] : (!fir.ref<!fir.type<_QMarray_derived_assignTsimple_copy{i:i32,c:!fir.array<20x!fir.char<1,10>>,p:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK:         %[[VAL_34:.*]] = fir.load %[[VAL_32]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK:         fir.store %[[VAL_34]] to %[[VAL_33]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
  ! CHECK:         %[[VAL_35:.*]] = arith.subi %[[VAL_9]], %[[VAL_6]] : index
  ! CHECK:         br ^bb1(%[[VAL_11]], %[[VAL_35]] : index, index)
  ! CHECK:       ^bb6:
  t1 = t2
  ! CHECK:         return
  ! CHECK:       }
end subroutine

! Types require more complex assignments are passed to the runtime
! CHECK-LABEL: func @_QMarray_derived_assignPtest_deep_copy(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<!fir.array<10x!fir.type<_QMarray_derived_assignTdeep_copy{i:i32,a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<!fir.array<10x!fir.type<_QMarray_derived_assignTdeep_copy{i:i32,a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>{{.*}}) {
subroutine test_deep_copy(t1, t2)
  ! CHECK-DAG:     %[[VAL_3:.*]] = arith.constant 10 : index
  ! CHECK-DAG:     %[[VAL_4:.*]] = arith.constant 0 : index
  ! CHECK-DAG:     %[[VAL_5:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_6:.*]] = fir.alloca !fir.box<!fir.type<_QMarray_derived_assignTdeep_copy{i:i32,a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
  ! CHECK:         %[[VAL_7:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
  ! CHECK:         br ^bb1(%[[VAL_4]], %[[VAL_3]] : index, index)
  ! CHECK:       ^bb1(%[[VAL_8:.*]]: index, %[[VAL_9:.*]]: index):
  ! CHECK:         %[[VAL_10:.*]] = arith.cmpi sgt, %[[VAL_9]], %[[VAL_4]] : index
  ! CHECK:         cond_br %[[VAL_10]], ^bb2, ^bb3
  ! CHECK:       ^bb2:
  ! CHECK:         %[[VAL_11:.*]] = arith.addi %[[VAL_8]], %[[VAL_5]] : index
  ! CHECK:         %[[VAL_12:.*]] = fir.array_coor %[[VAL_1]](%[[VAL_7]]) %[[VAL_11]] : (!fir.ref<!fir.array<10x!fir.type<_QMarray_derived_assignTdeep_copy{i:i32,a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QMarray_derived_assignTdeep_copy{i:i32,a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
  ! CHECK:         %[[VAL_13:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_7]]) %[[VAL_11]] : (!fir.ref<!fir.array<10x!fir.type<_QMarray_derived_assignTdeep_copy{i:i32,a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QMarray_derived_assignTdeep_copy{i:i32,a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
  ! CHECK:         %[[VAL_14:.*]] = fir.embox %[[VAL_13]] : (!fir.ref<!fir.type<_QMarray_derived_assignTdeep_copy{i:i32,a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> !fir.box<!fir.type<_QMarray_derived_assignTdeep_copy{i:i32,a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
  ! CHECK:         %[[VAL_15:.*]] = fir.embox %[[VAL_12]] : (!fir.ref<!fir.type<_QMarray_derived_assignTdeep_copy{i:i32,a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> !fir.box<!fir.type<_QMarray_derived_assignTdeep_copy{i:i32,a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>
  ! CHECK:         fir.store %[[VAL_14]] to %[[VAL_6]] : !fir.ref<!fir.box<!fir.type<_QMarray_derived_assignTdeep_copy{i:i32,a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>
  ! CHECK:         %[[VAL_16:.*]] = fir.address_of(@_QQcl.{{.*}}) : !fir.ref<!fir.char<1,
  ! CHECK:         %[[VAL_17:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<!fir.box<!fir.type<_QMarray_derived_assignTdeep_copy{i:i32,a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>) -> !fir.ref<!fir.box<none>>
  ! CHECK:         %[[VAL_18:.*]] = fir.convert %[[VAL_15]] : (!fir.box<!fir.type<_QMarray_derived_assignTdeep_copy{i:i32,a:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>) -> !fir.box<none>
  ! CHECK:         %[[VAL_19:.*]] = fir.convert %[[VAL_16]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
  ! CHECK:         %[[VAL_20:.*]] = fir.call @_FortranAAssign(%[[VAL_17]], %[[VAL_18]], %[[VAL_19]], %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32) -> none
  ! CHECK:         %[[VAL_21:.*]] = arith.subi %[[VAL_9]], %[[VAL_5]] : index
  ! CHECK:         br ^bb1(%[[VAL_11]], %[[VAL_21]] : index, index)
  type(deep_copy) :: t1(10), t2(10)
  t1 = t2
  ! CHECK:         return
  ! CHECK:       }
end subroutine
  
end module
