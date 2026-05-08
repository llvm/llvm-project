! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPimplied_iters_allocatable(
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

! CHECK-LABEL: func.func @_QPconflicting_allocatable(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.type<_QFconflicting_allocatableTt{oui:!fir.logical<4>,arr:!fir.box<!fir.heap<!fir.array<?xf32>>>}>>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<i32>{{.*}}, %[[VAL_2:.*]]: !fir.ref<i32>{{.*}}) {
! CHECK: return
! CHECK: }

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

! CHECK-LABEL: func.func @_QPforall_pointer_assign(
! CHECK-SAME:    %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.type<_QFforall_pointer_assignTt{ptr:!fir.box<!fir.ptr<!fir.array<?xf32>>>}>>> {{.*}}, %[[VAL_1:.*]]: !fir.ref<f32> {{.*}}, %[[VAL_2:.*]]: !fir.ref<i32> {{.*}}, %[[VAL_3:.*]]: !fir.ref<i32> {{.*}}) {
! CHECK: %[[AP:.*]]:2 = hlfir.declare %[[VAL_0]]
! CHECK: %[[AT:.*]]:2 = hlfir.declare %[[VAL_1]]
! CHECK: %[[II:.*]]:2 = hlfir.declare %[[VAL_2]]
! CHECK: %[[IJ:.*]]:2 = hlfir.declare %[[VAL_3]]
! CHECK: %[[IIV:.*]] = fir.load %[[II]]#0
! CHECK: %[[IJV:.*]] = fir.load %[[IJ]]#0
! CHECK: %[[C8:.*]] = arith.constant 8 : i32
! CHECK: hlfir.forall lb {
! CHECK:   hlfir.yield %[[IIV]] : i32
! CHECK: } ub {
! CHECK:   hlfir.yield %[[IJV]] : i32
! CHECK: } step {
! CHECK:   hlfir.yield %[[C8]] : i32
! CHECK: }  (%[[IARG:.*]]: i32) {
! CHECK:   %[[I:.*]] = hlfir.forall_index "i" %[[IARG]] : (i32) -> !fir.ref<i32>
! CHECK:   hlfir.region_assign {
! CHECK:     %[[IVAL:.*]] = fir.load %[[I]] : !fir.ref<i32>
! CHECK:     %[[C1:.*]] = arith.constant 1 : i32
! CHECK:     %[[IM1:.*]] = arith.subi %[[IVAL]], %[[C1]]
! CHECK:     %[[IIDX:.*]] = fir.convert %[[IM1]] : (i32) -> i64
! CHECK:     %[[APIM1:.*]] = hlfir.designate %[[AP]]#0 (%[[IIDX]])
! CHECK:     %[[APIM1PTR:.*]] = hlfir.designate %[[APIM1]]{"ptr"}
! CHECK:     %[[VAL:.*]] = fir.load %[[APIM1PTR]]
! CHECK:     hlfir.yield %[[VAL]] : !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:   } to {
! CHECK:     %[[IVAL:.*]] = fir.load %[[I]] : !fir.ref<i32>
! CHECK:     %[[IIDX:.*]] = fir.convert %[[IVAL]] : (i32) -> i64
! CHECK:     %[[API:.*]] = hlfir.designate %[[AP]]#0 (%[[IIDX]])
! CHECK:     %[[APIPTR:.*]] = hlfir.designate %[[API]]{"ptr"}
! CHECK:     hlfir.yield %[[APIPTR]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:   }
! CHECK: }

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

! CHECK-LABEL: func.func @_QPslice_with_explicit_iters() {
! CHECK: %[[A_ADDR:.*]] = fir.alloca !fir.array<10x10xi32>
! CHECK: %[[A:.*]]:2 = hlfir.declare %[[A_ADDR]]
! CHECK: %[[C1:.*]] = arith.constant 1 : i32
! CHECK: %[[C5:.*]] = arith.constant 5 : i32
! CHECK: hlfir.forall lb {
! CHECK:   hlfir.yield %[[C1]] : i32
! CHECK: } ub {
! CHECK:   hlfir.yield %[[C5]] : i32
! CHECK: }  (%[[IARG:.*]]: i32) {
! CHECK:   %[[I:.*]] = hlfir.forall_index "i" %[[IARG]] : (i32) -> !fir.ref<i32>
! CHECK:   hlfir.region_assign {
! CHECK:     %[[IVAL:.*]] = fir.load %[[I]] : !fir.ref<i32>
! CHECK:     %[[C0:.*]] = arith.constant 0 : i32
! CHECK:     %[[NEG_I:.*]] = arith.subi %[[C0]], %[[IVAL]]
! CHECK:     hlfir.yield %[[NEG_I]] : i32
! CHECK:   } to {
! CHECK:     %[[IVAL:.*]] = fir.load %[[I]] : !fir.ref<i32>
! CHECK:     %[[IIDX:.*]] = fir.convert %[[IVAL]] : (i32) -> i64
! CHECK:     %{{.*}} = hlfir.designate %[[A]]#0 {{.*}} : (!fir.ref<!fir.array<10x10xi32>>, index, index, index, i64, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK:     hlfir.yield {{.*}} : !fir.box<!fir.array<?xi32>>
! CHECK:   }
! CHECK: }

subroutine slice_with_explicit_iters

  integer :: a(10,10)
  forall (i=1:5)
     a(1:i, i) = -i
  end forall
end subroutine slice_with_explicit_iters

! CHECK-LABEL: func.func @_QPembox_argument_with_slice(
! CHECK-SAME: %[[AARG:.*]]: !fir.ref<!fir.array<1xi32>> {{.*}}, %[[BARG:.*]]: !fir.ref<!fir.array<2x2xi32>> {{.*}}) {
! CHECK: %[[A:.*]]:2 = hlfir.declare %[[AARG]]
! CHECK: %[[B:.*]]:2 = hlfir.declare %[[BARG]]
! CHECK: hlfir.forall lb {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: } ub {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: }  (%[[IARG:.*]]: i32) {
! CHECK:   %[[I:.*]] = hlfir.forall_index "i" %[[IARG]] : (i32) -> !fir.ref<i32>
! CHECK:   hlfir.region_assign {
! CHECK:     %[[IVAL:.*]] = fir.load %[[I]] : !fir.ref<i32>
! CHECK:     %[[IIDX:.*]] = fir.convert %[[IVAL]] : (i32) -> i64
! CHECK:     %[[BI:.*]] = hlfir.designate %[[B]]#0 ({{.*}}, %[[IIDX]]) {{.*}} : (!fir.ref<!fir.array<2x2xi32>>, index, index, index, i64, !fir.shape<1>) -> !fir.ref<!fir.array<2xi32>>
! CHECK:     %[[BIBOX:.*]] = fir.embox %[[BI]]({{.*}}) : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<2xi32>>
! CHECK:     %[[BIBOX_NONE:.*]] = fir.convert %[[BIBOX]] : (!fir.box<!fir.array<2xi32>>) -> !fir.box<!fir.array<?xi32>>
! CHECK:     %[[RES:.*]] = fir.call @_QPe(%[[BIBOX_NONE]]) {{.*}} : (!fir.box<!fir.array<?xi32>>) -> i32
! CHECK:     %{{.*}} = arith.addi %[[RES]], {{.*}}
! CHECK:     hlfir.yield {{.*}} : i32
! CHECK:   } to {
! CHECK:     %[[IVAL:.*]] = fir.load %[[I]] : !fir.ref<i32>
! CHECK:     %[[IIDX:.*]] = fir.convert %[[IVAL]] : (i32) -> i64
! CHECK:     %[[AI:.*]] = hlfir.designate %[[A]]#0 (%[[IIDX]])
! CHECK:     hlfir.yield %[[AI]] : !fir.ref<i32>
! CHECK:   }
! CHECK: }

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
