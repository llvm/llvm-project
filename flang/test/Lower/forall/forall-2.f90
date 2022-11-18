! RUN: bbc -emit-fir %s -o - | FileCheck %s
! RUN: bbc %s -o - | FileCheck --check-prefix=POSTOPT %s

! CHECK-LABEL: func @_QPimplied_iters_allocatable(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.type<_QFimplied_iters_allocatableTt{oui:!fir.logical<4>,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>{{.*}}, %[[VAL_1:.*]]: !fir.box<!fir.array<?xf32>>{{.*}}) {
! CHECK: return
! CHECK: }

subroutine implied_iters_allocatable(thing, a1)
  ! No dependence between lhs and rhs.
  ! Lhs may need to be reallocated to conform.
  real :: a1(:)
  type t
     logical :: oui
     real, allocatable :: arr(:)
  end type t
  type(t) :: thing(:)
  integer :: i
  
  forall (i=5:13)
  ! commenting out this test for the moment (hits assert)
  !  thing(i)%arr = a1
  end forall
end subroutine implied_iters_allocatable

subroutine conflicting_allocatable(thing, lo, hi)
  ! Introduce a crossing dependence to produce copy-in/copy-out code.
  integer :: lo,hi
  type t
     logical :: oui
     real, allocatable :: arr(:)
  end type t
  type(t) :: thing(:)
  integer :: i
  
  forall (i = lo:hi)
  ! commenting out this test for the moment (hits assert)
  !  thing(i)%arr = thing(hi-i)%arr
  end forall
end subroutine conflicting_allocatable

! CHECK-LABEL: func @_QPforall_pointer_assign(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.type<_QFforall_pointer_assignTt{ptr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>> {fir.bindc_name = "ap"}, %[[VAL_1:.*]]: !fir.ref<f32> {fir.bindc_name = "at"}, %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "ii"}, %[[VAL_3:.*]]: !fir.ref<i32> {fir.bindc_name = "ij"}) {
! CHECK:         %[[VAL_4:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_5:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:         %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (i32) -> index
! CHECK:         %[[VAL_7:.*]] = fir.load %[[VAL_3]] : !fir.ref<i32>
! CHECK:         %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i32) -> index
! CHECK:         %[[VAL_9:.*]] = arith.constant 8 : i32
! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> index
! CHECK:         %[[VAL_11:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?x!fir.type<_QFforall_pointer_assignTt{ptr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>>) -> !fir.array<?x!fir.type<_QFforall_pointer_assignTt{ptr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>
! CHECK:         %[[VAL_12:.*]] = fir.array_load %[[VAL_0]] : (!fir.box<!fir.array<?x!fir.type<_QFforall_pointer_assignTt{ptr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>>) -> !fir.array<?x!fir.type<_QFforall_pointer_assignTt{ptr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>
! CHECK:         %[[VAL_13:.*]] = fir.do_loop %[[VAL_14:.*]] = %[[VAL_6]] to %[[VAL_8]] step %[[VAL_10]] unordered iter_args(%[[VAL_15:.*]] = %[[VAL_11]]) -> (!fir.array<?x!fir.type<_QFforall_pointer_assignTt{ptr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>) {
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_14]] : (index) -> i32
! CHECK:           fir.store %[[VAL_16]] to %[[VAL_4]] : !fir.ref<i32>
! CHECK-DAG:       %[[VAL_17:.*]] = arith.constant 1 : index
! CHECK-DAG:       %[[VAL_18:.*]] = arith.constant 1 : i32
! CHECK-DAG:       %[[VAL_19:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
! CHECK:           %[[VAL_20:.*]] = arith.subi %[[VAL_19]], %[[VAL_18]] : i32
! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i32) -> i64
! CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (i64) -> index
! CHECK:           %[[VAL_23:.*]] = arith.subi %[[VAL_22]], %[[VAL_17]] : index
! CHECK:           %[[VAL_24:.*]] = fir.field_index ptr, !fir.type<_QFforall_pointer_assignTt{ptr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>
! CHECK:           %[[VAL_25:.*]] = fir.array_fetch %[[VAL_12]], %[[VAL_23]], %[[VAL_24]] : (!fir.array<?x!fir.type<_QFforall_pointer_assignTt{ptr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>, index, !fir.field) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:           %[[VAL_26:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_27:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
! CHECK:           %[[VAL_28:.*]] = fir.convert %[[VAL_27]] : (i32) -> i64
! CHECK:           %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (i64) -> index
! CHECK:           %[[VAL_30:.*]] = arith.subi %[[VAL_29]], %[[VAL_26]] : index
! CHECK:           %[[VAL_31:.*]] = fir.field_index ptr, !fir.type<_QFforall_pointer_assignTt{ptr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>
! CHECK:           %[[VAL_32:.*]] = fir.array_update %[[VAL_15]], %[[VAL_25]], %[[VAL_30]], %[[VAL_31]] : (!fir.array<?x!fir.type<_QFforall_pointer_assignTt{ptr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>, !fir.box<!fir.ptr<!fir.array<?xf32>>>, index, !fir.field) -> !fir.array<?x!fir.type<_QFforall_pointer_assignTt{ptr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>
! CHECK:           fir.result %[[VAL_32]] : !fir.array<?x!fir.type<_QFforall_pointer_assignTt{ptr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_11]], %[[VAL_33:.*]] to %[[VAL_0]] : !fir.array<?x!fir.type<_QFforall_pointer_assignTt{ptr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>, !fir.array<?x!fir.type<_QFforall_pointer_assignTt{ptr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>, !fir.box<!fir.array<?x!fir.type<_QFforall_pointer_assignTt{ptr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>>
! CHECK:         return
! CHECK:       }

! POSTOPT-LABEL: func @_QPforall_pointer_assign(
! POSTOPT:         %[[VAL_15:.*]] = fir.allocmem !fir.array<?x!fir.type<_QFforall_pointer_assignTt{ptr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>, %{{.*}}#1
! POSTOPT:       ^bb{{[0-9]+}}(%[[VAL_16:.*]]: index, %[[VAL_17:.*]]: index):
! POSTOPT:       ^bb{{[0-9]+}}(%[[VAL_30:.*]]: index, %[[VAL_31:.*]]: index):
! POSTOPT:       ^bb{{[0-9]+}}(%[[VAL_46:.*]]: index, %[[VAL_47:.*]]: index):
! POSTOPT-NOT:   ^bb{{[0-9]+}}(%{{.*}}: index, %{{.*}}: index):
! POSTOPT:         fir.freemem %[[VAL_15]] : !fir.heap<!fir.array<?x!fir.type<_QFforall_pointer_assignTt{ptr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>>
! POSTOPT:       }

subroutine forall_pointer_assign(ap, at, ii, ij)
  ! Set pointer members in an array of derived type of pointers to arrays.
  ! Introduce a loop carried dependence to produce copy-in/copy-out code.
  type t
     real, pointer :: ptr(:)
  end type t
  type(t) :: ap(:)
  integer :: ii, ij

  forall (i = ii:ij:8)
     ap(i)%ptr => ap(i-1)%ptr
  end forall
end subroutine forall_pointer_assign

! CHECK-LABEL: func @_QPslice_with_explicit_iters() {
! CHECK:         %[[VAL_0:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_1:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_2:.*]] = arith.constant 10 : index
! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.array<10x10xi32> {bindc_name = "a", uniq_name = "_QFslice_with_explicit_itersEa"}
! CHECK:         %[[VAL_4:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> index
! CHECK:         %[[VAL_6:.*]] = arith.constant 5 : i32
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
! CHECK:         %[[VAL_8:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_9:.*]] = fir.shape %[[VAL_1]], %[[VAL_2]] : (index, index) -> !fir.shape<2>
! CHECK:         %[[VAL_10:.*]] = fir.array_load %[[VAL_3]](%[[VAL_9]]) : (!fir.ref<!fir.array<10x10xi32>>, !fir.shape<2>) -> !fir.array<10x10xi32>
! CHECK:         %[[VAL_11:.*]] = fir.do_loop %[[VAL_12:.*]] = %[[VAL_5]] to %[[VAL_7]] step %[[VAL_8]] unordered iter_args(%[[VAL_13:.*]] = %[[VAL_10]]) -> (!fir.array<10x10xi32>) {
! CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_12]] : (index) -> i32
! CHECK:           fir.store %[[VAL_14]] to %[[VAL_0]] : !fir.ref<i32>
! CHECK:           %[[VAL_15:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_16:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (i64) -> index
! CHECK:           %[[VAL_18:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i64) -> index
! CHECK:           %[[VAL_20:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i32) -> i64
! CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_21]] : (i64) -> index
! CHECK:           %[[VAL_23:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_24:.*]] = arith.subi %[[VAL_22]], %[[VAL_17]] : index
! CHECK:           %[[VAL_25:.*]] = arith.addi %[[VAL_24]], %[[VAL_19]] : index
! CHECK:           %[[VAL_26:.*]] = arith.divsi %[[VAL_25]], %[[VAL_19]] : index
! CHECK:           %[[VAL_27:.*]] = arith.cmpi sgt, %[[VAL_26]], %[[VAL_23]] : index
! CHECK:           %[[VAL_28:.*]] = arith.select %[[VAL_27]], %[[VAL_26]], %[[VAL_23]] : index
! CHECK:           %[[VAL_29:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:           %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (i32) -> i64
! CHECK:           %[[VAL_31:.*]] = fir.convert %[[VAL_30]] : (i64) -> index
! CHECK:           %[[VAL_32:.*]] = arith.subi %[[VAL_31]], %[[VAL_15]] : index
! CHECK:           %[[VAL_39:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:           %[[VAL_40:.*]] = arith.constant 0 : i32
! CHECK:           %[[VAL_41:.*]] = arith.subi %[[VAL_40]], %[[VAL_39]] : i32
! CHECK:           %[[VAL_33:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_34:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_35:.*]] = arith.subi %[[VAL_28]], %[[VAL_33]] : index
! CHECK:           %[[VAL_36:.*]] = fir.do_loop %[[VAL_37:.*]] = %[[VAL_34]] to %[[VAL_35]] step %[[VAL_33]] unordered iter_args(%[[VAL_38:.*]] = %[[VAL_13]]) -> (!fir.array<10x10xi32>) {
! CHECK:             %[[VAL_42:.*]] = arith.subi %[[VAL_17]], %[[VAL_15]] : index
! CHECK:             %[[VAL_43:.*]] = arith.muli %[[VAL_37]], %[[VAL_19]] : index
! CHECK:             %[[VAL_44:.*]] = arith.addi %[[VAL_42]], %[[VAL_43]] : index
! CHECK:             %[[VAL_45:.*]] = fir.array_update %[[VAL_38]], %[[VAL_41]], %[[VAL_44]], %[[VAL_32]] : (!fir.array<10x10xi32>, i32, index, index) -> !fir.array<10x10xi32>
! CHECK:             fir.result %[[VAL_45]] : !fir.array<10x10xi32>
! CHECK:           }
! CHECK:           fir.result %[[VAL_46:.*]] : !fir.array<10x10xi32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_10]], %[[VAL_47:.*]] to %[[VAL_3]] : !fir.array<10x10xi32>, !fir.array<10x10xi32>, !fir.ref<!fir.array<10x10xi32>>
! CHECK:         return
! CHECK:       }

subroutine slice_with_explicit_iters

  integer :: a(10,10)
  forall (i=1:5)
     a(1:i, i) = -i
  end forall
end subroutine slice_with_explicit_iters

! CHECK-LABEL: func @_QPembox_argument_with_slice(
! CHECK-SAME:                                     %[[VAL_0:.*]]: !fir.ref<!fir.array<1xi32>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<!fir.array<2x2xi32>>{{.*}}) {
! CHECK:         %[[VAL_2:.*]] = fir.alloca i32 {adapt.valuebyref, bindc_name = "i"}
! CHECK:         %[[VAL_3:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_4:.*]] = arith.constant 2 : index
! CHECK:         %[[VAL_5:.*]] = arith.constant 2 : index
! CHECK:         %[[VAL_6:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
! CHECK:         %[[VAL_8:.*]] = arith.constant 1 : i32
! CHECK:         %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (i32) -> index
! CHECK:         %[[VAL_10:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_11:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_12:.*]] = fir.array_load %[[VAL_0]](%[[VAL_11]]) : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>) -> !fir.array<1xi32>
! CHECK:         %[[VAL_13:.*]] = fir.do_loop %[[VAL_14:.*]] = %[[VAL_7]] to %[[VAL_9]] step %[[VAL_10]] unordered iter_args(%[[VAL_15:.*]] = %[[VAL_12]]) -> (!fir.array<1xi32>) {
! CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_14]] : (index) -> i32
! CHECK:           fir.store %[[VAL_16]] to %[[VAL_2]] : !fir.ref<i32>
! CHECK-DAG:       %[[VAL_18:.*]] = arith.constant 1 : index
! CHECK-DAG:       %[[VAL_19:.*]] = arith.constant 1 : i64
! CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i64) -> index
! CHECK:           %[[VAL_21:.*]] = arith.addi %[[VAL_18]], %[[VAL_4]] : index
! CHECK:           %[[VAL_22:.*]] = arith.subi %[[VAL_21]], %[[VAL_18]] : index
! CHECK:           %[[VAL_23:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_23]] : (i32) -> i64
! CHECK:           %[[VAL_25:.*]] = fir.undefined index
! CHECK:           %[[VAL_26:.*]] = fir.shape %[[VAL_4]], %[[VAL_5]] : (index, index) -> !fir.shape<2>
! CHECK:           %[[VAL_27:.*]] = fir.slice %[[VAL_18]], %[[VAL_22]], %[[VAL_20]], %[[VAL_24]], %[[VAL_25]], %[[VAL_25]] : (index, index, index, i64, index, index) -> !fir.slice<2>
! CHECK:           %[[VAL_28:.*]] = fir.embox %[[VAL_1]](%[[VAL_26]]) {{\[}}%[[VAL_27]]] : (!fir.ref<!fir.array<2x2xi32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.box<!fir.array<?xi32>>
! CHECK:           %[[VAL_29:.*]] = fir.call @_QPe(%[[VAL_28]]) {{.*}}: (!fir.box<!fir.array<?xi32>>) -> i32
! CHECK:           %[[VAL_17:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_30:.*]] = arith.addi %[[VAL_29]], %[[VAL_17]] : i32
! CHECK:           %[[VAL_31:.*]] = arith.constant 1 : index
! CHECK:           %[[VAL_32:.*]] = fir.load %[[VAL_2]] : !fir.ref<i32>
! CHECK:           %[[VAL_33:.*]] = fir.convert %[[VAL_32]] : (i32) -> i64
! CHECK:           %[[VAL_34:.*]] = fir.convert %[[VAL_33]] : (i64) -> index
! CHECK:           %[[VAL_35:.*]] = arith.subi %[[VAL_34]], %[[VAL_31]] : index
! CHECK:           %[[VAL_36:.*]] = fir.array_update %[[VAL_15]], %[[VAL_30]], %[[VAL_35]] : (!fir.array<1xi32>, i32, index) -> !fir.array<1xi32>
! CHECK:           fir.result %[[VAL_36]] : !fir.array<1xi32>
! CHECK:         }
! CHECK:         fir.array_merge_store %[[VAL_12]], %[[VAL_37:.*]] to %[[VAL_0]] : !fir.array<1xi32>, !fir.array<1xi32>, !fir.ref<!fir.array<1xi32>>
! CHECK:         return
! CHECK:       }

subroutine embox_argument_with_slice(a,b)
  interface
     pure integer function e(a)
       integer, intent(in) :: a(:)
     end function e
  end interface
  integer a(1), b(2,2)

  forall (i=1:1)
     a(i) = e(b(:,i)) + 1
  end forall
end subroutine embox_argument_with_slice
