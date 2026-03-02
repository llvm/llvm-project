! Test lowering of arrays of POINTER.
!
! An array of pointer to T can be constructed by having an array of
! derived type, where the derived type has a pointer to T
! component. An entity with both the DIMENSION and POINTER attributes
! is a pointer to an array of T and never an array of pointer to T in
! Fortran.

! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

module array_of_pointer_test
  type t
     integer, POINTER :: ip
  end type t

  type u
     integer :: v
  end type u

  type tu
     type(u), POINTER :: ip
  end type tu

  type ta
     integer, POINTER :: ip(:)
  end type ta

  type tb
     integer, POINTER :: ip(:,:)
  end type tb

  type tv
     type(tu), POINTER :: jp(:)
  end type tv

  ! Derived types with type parameters hit a TODO.
!  type ct(l)
!     integer, len :: l
!     character(LEN=l), POINTER :: cp
!  end type ct

!  type cu(l)
!     integer, len :: l
!     character(LEN=l) :: cv
!  end type cu
end module array_of_pointer_test

subroutine s1(x,y)
  use array_of_pointer_test
  type(t) :: x(:)
  integer :: y(:)

  forall (i=1:10)
     ! assign value to pointee variable
     x(i)%ip = y(i)
  end forall
end subroutine s1

! CHECK-LABEL: func.func @_QPs1(
! CHECK-SAME: %[[XARG:.*]]: !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>> {{.*}}, %[[YARG:.*]]: !fir.box<!fir.array<?xi32>> {{.*}}) {
! CHECK: %[[X:.*]]:2 = hlfir.declare %[[XARG]]
! CHECK: %[[Y:.*]]:2 = hlfir.declare %[[YARG]]
! CHECK: hlfir.forall lb {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: } ub {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: }  (%[[IARG:.*]]: i32) {
! CHECK:   %[[I:.*]] = hlfir.forall_index "i" %[[IARG]] : (i32) -> !fir.ref<i32>
! CHECK:   hlfir.region_assign {
! CHECK:     %[[IVAL:.*]] = fir.load %[[I]] : !fir.ref<i32>
! CHECK:     %[[IIDX:.*]] = fir.convert %[[IVAL]] : (i32) -> i64
! CHECK:     %[[YI:.*]] = hlfir.designate %[[Y]]#0 (%[[IIDX]])
! CHECK:     %[[YIVAL:.*]] = fir.load %[[YI]] : !fir.ref<i32>
! CHECK:     hlfir.yield %[[YIVAL]] : i32
! CHECK:   } to {
! CHECK:     %[[IVAL:.*]] = fir.load %[[I]] : !fir.ref<i32>
! CHECK:     %[[IIDX:.*]] = fir.convert %[[IVAL]] : (i32) -> i64
! CHECK:     %[[XI:.*]] = hlfir.designate %[[X]]#0 (%[[IIDX]])
! CHECK:     %[[XIIP:.*]] = hlfir.designate %[[XI]]{"ip"}
! CHECK:     %[[IPBOX:.*]] = fir.load %[[XIIP]]
! CHECK:     %[[IPADDR:.*]] = fir.box_addr %[[IPBOX]]
! CHECK:     hlfir.yield %[[IPADDR]] : !fir.ptr<i32>
! CHECK:   }
! CHECK: }

subroutine s1_1(x,y)
  use array_of_pointer_test
  type(t) :: x(10)
  integer :: y(10)

  forall (i=1:10)
     ! assign value to pointee variable
     x(i)%ip = y(i)
  end forall
end subroutine s1_1

! CHECK-LABEL: func.func @_QPs1_1(
! CHECK-SAME: %[[XARG:.*]]: !fir.ref<!fir.array<10x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>> {{.*}}, %[[YARG:.*]]: !fir.ref<!fir.array<10xi32>> {{.*}}) {
! CHECK: %[[X:.*]]:2 = hlfir.declare %[[XARG]]
! CHECK: %[[Y:.*]]:2 = hlfir.declare %[[YARG]]
! CHECK: hlfir.forall lb {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: } ub {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: }  (%[[IARG:.*]]: i32) {
! CHECK:   %[[I:.*]] = hlfir.forall_index "i" %[[IARG]] : (i32) -> !fir.ref<i32>
! CHECK:   hlfir.region_assign {
! CHECK:     %[[IVAL:.*]] = fir.load %[[I]] : !fir.ref<i32>
! CHECK:     %[[IIDX:.*]] = fir.convert %[[IVAL]] : (i32) -> i64
! CHECK:     %[[YI:.*]] = hlfir.designate %[[Y]]#0 (%[[IIDX]])
! CHECK:     %[[YIVAL:.*]] = fir.load %[[YI]] : !fir.ref<i32>
! CHECK:     hlfir.yield %[[YIVAL]] : i32
! CHECK:   } to {
! CHECK:     %[[IVAL:.*]] = fir.load %[[I]] : !fir.ref<i32>
! CHECK:     %[[IIDX:.*]] = fir.convert %[[IVAL]] : (i32) -> i64
! CHECK:     %[[XI:.*]] = hlfir.designate %[[X]]#0 (%[[IIDX]])
! CHECK:     %[[XIIP:.*]] = hlfir.designate %[[XI]]{"ip"}
! CHECK:     %[[IPBOX:.*]] = fir.load %[[XIIP]]
! CHECK:     %[[IPADDR:.*]] = fir.box_addr %[[IPBOX]]
! CHECK:     hlfir.yield %[[IPADDR]] : !fir.ptr<i32>
! CHECK:   }
! CHECK: }

! Dependent type assignment, TODO
!subroutine s1_2(x,y,l)
!  use array_of_pointer_test
!  type(ct(l)) :: x(10)
!  character(l) :: y(10)

!  forall (i=1:10)
     ! assign value to pointee variable
!     x(i)%cp = y(i)
!  end forall
!end subroutine s1_2

subroutine s2(x,y)
  use array_of_pointer_test
  type(t) :: x(:)
  integer, TARGET :: y(:)

  forall (i=1:10)
     ! assign address to POINTER
     x(i)%ip => y(i)
  end forall
end subroutine s2

! CHECK-LABEL: func.func @_QPs2(
! CHECK-SAME: %[[XARG:.*]]: !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>> {{.*}}, %[[YARG:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "y", fir.target}) {
! CHECK: %[[X:.*]]:2 = hlfir.declare %[[XARG]]
! CHECK: %[[Y:.*]]:2 = hlfir.declare %[[YARG]]
! CHECK: hlfir.forall lb {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: } ub {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: }  (%[[IARG:.*]]: i32) {
! CHECK:   %[[I:.*]] = hlfir.forall_index "i" %[[IARG]] : (i32) -> !fir.ref<i32>
! CHECK:   hlfir.region_assign {
! CHECK:     %[[IVAL:.*]] = fir.load %[[I]] : !fir.ref<i32>
! CHECK:     %[[IIDX:.*]] = fir.convert %[[IVAL]] : (i32) -> i64
! CHECK:     %[[YI:.*]] = hlfir.designate %[[Y]]#0 (%[[IIDX]])
! CHECK:     %[[YIBOX:.*]] = fir.embox %[[YI]] : (!fir.ref<i32>) -> !fir.box<!fir.ptr<i32>>
! CHECK:     hlfir.yield %[[YIBOX]] : !fir.box<!fir.ptr<i32>>
! CHECK:   } to {
! CHECK:     %[[IVAL:.*]] = fir.load %[[I]] : !fir.ref<i32>
! CHECK:     %[[IIDX:.*]] = fir.convert %[[IVAL]] : (i32) -> i64
! CHECK:     %[[XI:.*]] = hlfir.designate %[[X]]#0 (%[[IIDX]])
! CHECK:     %[[XIIP:.*]] = hlfir.designate %[[XI]]{"ip"}
! CHECK:     hlfir.yield %[[XIIP]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:   }
! CHECK: }

subroutine s2_1(x,y)
  use array_of_pointer_test
  type(t) :: x(:)
  integer, POINTER :: y(:)

  forall (i=1:10)
     ! assign address to POINTER
     x(i)%ip => y(i)
  end forall
end subroutine s2_1

! CHECK-LABEL: func.func @_QPs2_1(
! CHECK-SAME: %[[XARG:.*]]: !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>> {{.*}}, %[[YARG:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {{.*}}) {
! CHECK: %[[X:.*]]:2 = hlfir.declare %[[XARG]]
! CHECK: %[[Y:.*]]:2 = hlfir.declare %[[YARG]]
! CHECK: hlfir.forall lb {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: } ub {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: }  (%[[IARG:.*]]: i32) {
! CHECK:   %[[I:.*]] = hlfir.forall_index "i" %[[IARG]] : (i32) -> !fir.ref<i32>
! CHECK:   hlfir.region_assign {
! CHECK:     %[[YBOX:.*]] = fir.load %[[Y]]#0
! CHECK:     %[[IVAL:.*]] = fir.load %[[I]] : !fir.ref<i32>
! CHECK:     %[[IIDX:.*]] = fir.convert %[[IVAL]] : (i32) -> i64
! CHECK:     %[[YI:.*]] = hlfir.designate %[[YBOX]] (%[[IIDX]])
! CHECK:     %[[YIBOX:.*]] = fir.embox %[[YI]] : (!fir.ref<i32>) -> !fir.box<!fir.ptr<i32>>
! CHECK:     hlfir.yield %[[YIBOX]] : !fir.box<!fir.ptr<i32>>
! CHECK:   } to {
! CHECK:     %[[IVAL:.*]] = fir.load %[[I]] : !fir.ref<i32>
! CHECK:     %[[IIDX:.*]] = fir.convert %[[IVAL]] : (i32) -> i64
! CHECK:     %[[XI:.*]] = hlfir.designate %[[X]]#0 (%[[IIDX]])
! CHECK:     %[[XIIP:.*]] = hlfir.designate %[[XI]]{"ip"}
! CHECK:     hlfir.yield %[[XIIP]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:   }
! CHECK: }

subroutine s2_2(x,y)
  use array_of_pointer_test
  type(t) :: x(:)
  integer, ALLOCATABLE, TARGET :: y(:)

  forall (i=1:10)
     ! assign address to POINTER
     x(i)%ip => y(i)
  end forall
end subroutine s2_2

! CHECK-LABEL: func.func @_QPs2_2(
! CHECK-SAME: %[[XARG:.*]]: !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>> {{.*}}, %[[YARG:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {{.*}}) {
! CHECK: %[[X:.*]]:2 = hlfir.declare %[[XARG]]
! CHECK: %[[Y:.*]]:2 = hlfir.declare %[[YARG]]
! CHECK: hlfir.forall lb {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: } ub {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: }  (%[[IARG:.*]]: i32) {
! CHECK:   %[[I:.*]] = hlfir.forall_index "i" %[[IARG]] : (i32) -> !fir.ref<i32>
! CHECK:   hlfir.region_assign {
! CHECK:     %[[YBOX:.*]] = fir.load %[[Y]]#0
! CHECK:     %[[IVAL:.*]] = fir.load %[[I]] : !fir.ref<i32>
! CHECK:     %[[IIDX:.*]] = fir.convert %[[IVAL]] : (i32) -> i64
! CHECK:     %[[YI:.*]] = hlfir.designate %[[YBOX]] (%[[IIDX]])
! CHECK:     %[[YIBOX:.*]] = fir.embox %[[YI]] : (!fir.ref<i32>) -> !fir.box<!fir.ptr<i32>>
! CHECK:     hlfir.yield %[[YIBOX]] : !fir.box<!fir.ptr<i32>>
! CHECK:   } to {
! CHECK:     %[[IVAL:.*]] = fir.load %[[I]] : !fir.ref<i32>
! CHECK:     %[[IIDX:.*]] = fir.convert %[[IVAL]] : (i32) -> i64
! CHECK:     %[[XI:.*]] = hlfir.designate %[[X]]#0 (%[[IIDX]])
! CHECK:     %[[XIIP:.*]] = hlfir.designate %[[XI]]{"ip"}
! CHECK:     hlfir.yield %[[XIIP]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:   }
! CHECK: }

subroutine s2_3(x)
  use array_of_pointer_test
  type(t) :: x(:)
  ! This is legal, but a bad idea.
  integer, ALLOCATABLE, TARGET :: y(:)

  forall (i=1:10)
     ! assign address to POINTER
     x(i)%ip => y(i)
  end forall
  ! x's pointers will remain associated, and may point to deallocated y.
end subroutine s2_3

! CHECK-LABEL: func.func @_QPs2_3(
! CHECK-SAME: %[[XARG:.*]]: !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>> {{.*}}) {
! CHECK: %[[X:.*]]:2 = hlfir.declare %[[XARG]]
! CHECK: %[[YALLOC:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK: %[[Y:.*]]:2 = hlfir.declare %[[YALLOC]]
! CHECK: hlfir.forall lb {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: } ub {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: }  (%[[IARG:.*]]: i32) {
! CHECK:   %[[I:.*]] = hlfir.forall_index "i" %[[IARG]] : (i32) -> !fir.ref<i32>
! CHECK:   hlfir.region_assign {
! CHECK:     %[[YBOX:.*]] = fir.load %[[Y]]#0
! CHECK:     %[[IVAL:.*]] = fir.load %[[I]] : !fir.ref<i32>
! CHECK:     %[[IIDX:.*]] = fir.convert %[[IVAL]] : (i32) -> i64
! CHECK:     %[[YI:.*]] = hlfir.designate %[[YBOX]] (%[[IIDX]])
! CHECK:     %[[YIBOX:.*]] = fir.embox %[[YI]] : (!fir.ref<i32>) -> !fir.box<!fir.ptr<i32>>
! CHECK:     hlfir.yield %[[YIBOX]] : !fir.box<!fir.ptr<i32>>
! CHECK:   } to {
! CHECK:     %[[IVAL:.*]] = fir.load %[[I]] : !fir.ref<i32>
! CHECK:     %[[IIDX:.*]] = fir.convert %[[IVAL]] : (i32) -> i64
! CHECK:     %[[XI:.*]] = hlfir.designate %[[X]]#0 (%[[IIDX]])
! CHECK:     %[[XIIP:.*]] = hlfir.designate %[[XI]]{"ip"}
! CHECK:     hlfir.yield %[[XIIP]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:   }
! CHECK: }

! Dependent type - TODO
!subroutine s2_4(x,y,l)
!  use array_of_pointer_test
!  type(ct(l)) :: x(:)
!  character(l), TARGET :: y(:)

!  forall (i=1:10)
     ! assign address to POINTER
!     x(i)%cp => y(i)
!  end forall
!end subroutine s2_4

subroutine s3(x,y)
  use array_of_pointer_test
  type(tu) :: x(:)
  integer :: y(:)

  forall (i=1:10)
     ! assign value to variable, indirecting through box
     x(i)%ip%v = y(i)
  end forall
end subroutine s3

! CHECK-LABEL: func.func @_QPs3(
! CHECK-SAME: %[[XARG:.*]]: !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{{.*}}>>> {{.*}}, %[[YARG:.*]]: !fir.box<!fir.array<?xi32>> {{.*}}) {
! CHECK: %[[X:.*]]:2 = hlfir.declare %[[XARG]]
! CHECK: %[[Y:.*]]:2 = hlfir.declare %[[YARG]]
! CHECK: hlfir.forall lb {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: } ub {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: }  (%[[IARG:.*]]: i32) {
! CHECK:   %[[I:.*]] = hlfir.forall_index "i" %[[IARG]] : (i32) -> !fir.ref<i32>
! CHECK:   hlfir.region_assign {
! CHECK:     %[[IVAL:.*]] = fir.load %[[I]] : !fir.ref<i32>
! CHECK:     %[[IIDX:.*]] = fir.convert %[[IVAL]] : (i32) -> i64
! CHECK:     %[[YI:.*]] = hlfir.designate %[[Y]]#0 (%[[IIDX]])
! CHECK:     %[[YIVAL:.*]] = fir.load %[[YI]] : !fir.ref<i32>
! CHECK:     hlfir.yield %[[YIVAL]] : i32
! CHECK:   } to {
! CHECK:     %[[IVAL:.*]] = fir.load %[[I]] : !fir.ref<i32>
! CHECK:     %[[IIDX:.*]] = fir.convert %[[IVAL]] : (i32) -> i64
! CHECK:     %[[XI:.*]] = hlfir.designate %[[X]]#0 (%[[IIDX]])
! CHECK:     %[[XIIP:.*]] = hlfir.designate %[[XI]]{"ip"}
! CHECK:     %[[IPBOX:.*]] = fir.load %[[XIIP]]
! CHECK:     %[[IPADDR:.*]] = fir.box_addr %[[IPBOX]]
! CHECK:     %[[V:.*]] = hlfir.designate %[[IPADDR]]{"v"}
! CHECK:     hlfir.yield %[[V]] : !fir.ref<i32>
! CHECK:   }
! CHECK: }

subroutine s3_1(x,y)
  use array_of_pointer_test
  type(tu) :: x(:)
  integer :: y(:)

  forall (i=1:10)
     ! assign value to variable, indirecting through box
     x(i)%ip%v = y(i)
  end forall
end subroutine s3_1

! CHECK-LABEL: func.func @_QPs3_1(
! CHECK-SAME: %[[XARG:.*]]: !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTtu{{.*}}>>> {{.*}}, %[[YARG:.*]]: !fir.box<!fir.array<?xi32>> {{.*}}) {
! CHECK: %[[X:.*]]:2 = hlfir.declare %[[XARG]]
! CHECK: %[[Y:.*]]:2 = hlfir.declare %[[YARG]]
! CHECK: hlfir.forall lb {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: } ub {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: }  (%[[IARG:.*]]: i32) {
! CHECK:   %[[I:.*]] = hlfir.forall_index "i" %[[IARG]] : (i32) -> !fir.ref<i32>
! CHECK:   hlfir.region_assign {
! CHECK:     %[[IVAL:.*]] = fir.load %[[I]] : !fir.ref<i32>
! CHECK:     %[[IIDX:.*]] = fir.convert %[[IVAL]] : (i32) -> i64
! CHECK:     %[[YI:.*]] = hlfir.designate %[[Y]]#0 (%[[IIDX]])
! CHECK:     %[[YIVAL:.*]] = fir.load %[[YI]] : !fir.ref<i32>
! CHECK:     hlfir.yield %[[YIVAL]] : i32
! CHECK:   } to {
! CHECK:     %[[IVAL:.*]] = fir.load %[[I]] : !fir.ref<i32>
! CHECK:     %[[IIDX:.*]] = fir.convert %[[IVAL]] : (i32) -> i64
! CHECK:     %[[XI:.*]] = hlfir.designate %[[X]]#0 (%[[IIDX]])
! CHECK:     %[[XIIP:.*]] = hlfir.designate %[[XI]]{"ip"}
! CHECK:     %[[IPBOX:.*]] = fir.load %[[XIIP]]
! CHECK:     %[[IPADDR:.*]] = fir.box_addr %[[IPBOX]]
! CHECK:     %[[V:.*]] = hlfir.designate %[[IPADDR]]{"v"}
! CHECK:     hlfir.yield %[[V]] : !fir.ref<i32>
! CHECK:   }
! CHECK: }

! Slice a target array and assign the box to a pointer of rank-1 field.
! RHS is an array section. Hits a TODO.
subroutine s4(x,y)
  use array_of_pointer_test
  type(ta) :: x(:)
  integer, TARGET :: y(:)

  forall (i=1:10)
     ! TODO: auto boxing of ranked RHS
!    x(i)%ip => y(i:i+1)
  end forall
end subroutine s4

! CHECK-LABEL: func.func @_QPs4(
! CHECK: hlfir.forall lb {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: } ub {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: }  (%[[IARG:.*]]: i32) {
! CHECK: }

! Most other Fortran implementations cannot compile the following 2 cases, s5
! and s5_1.
subroutine s5(x,y,z,n1,n2)
  use array_of_pointer_test
  type(ta) :: x(:)
  type(tb) :: y(:)
  type(ta), TARGET :: z(:)

  forall (i=1:10)
     ! Convert the rank-1 array to a rank-2 array on assignment
     y(i)%ip(1:n1,1:n2) => z(i)%ip
  end forall
end subroutine s5

! CHECK-LABEL: func.func @_QPs5(
! CHECK-SAME: %[[XARG:.*]]: !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTta{{.*}}>>> {{.*}}, %[[YARG:.*]]: !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTtb{{.*}}>>> {{.*}}, %[[ZARG:.*]]: !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTta{{.*}}>>> {{.*}}, %[[N1ARG:.*]]: !fir.ref<i32> {{.*}}, %[[N2ARG:.*]]: !fir.ref<i32> {{.*}}) {
! CHECK: %[[N1:.*]]:2 = hlfir.declare %[[N1ARG]]
! CHECK: %[[N2:.*]]:2 = hlfir.declare %[[N2ARG]]
! CHECK: %[[Y:.*]]:2 = hlfir.declare %[[YARG]]
! CHECK: %[[Z:.*]]:2 = hlfir.declare %[[ZARG]]
! CHECK: hlfir.forall lb {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: } ub {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: }  (%[[IARG:.*]]: i32) {
! CHECK:   %[[I:.*]] = hlfir.forall_index "i" %[[IARG]] : (i32) -> !fir.ref<i32>
! CHECK:   hlfir.region_assign {
! CHECK:     %[[IVAL:.*]] = fir.load %[[I]] : !fir.ref<i32>
! CHECK:     %[[IIDX:.*]] = fir.convert %[[IVAL]] : (i32) -> i64
! CHECK:     %[[ZI:.*]] = hlfir.designate %[[Z]]#0 (%[[IIDX]])
! CHECK:     %[[ZIIP:.*]] = hlfir.designate %[[ZI]]{"ip"}
! CHECK:     %[[IPBOX:.*]] = fir.load %[[ZIIP]]
! CHECK:     %{{.*}} = fir.shape_shift {{.*}} : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:     %[[REBOX:.*]] = fir.rebox %[[IPBOX]]({{.*}})
! CHECK:     hlfir.yield %[[REBOX]] : !fir.box<!fir.ptr<!fir.array<?x?xi32>>>
! CHECK:   } to {
! CHECK:     %[[IVAL:.*]] = fir.load %[[I]] : !fir.ref<i32>
! CHECK:     %[[IIDX:.*]] = fir.convert %[[IVAL]] : (i32) -> i64
! CHECK:     %[[YI:.*]] = hlfir.designate %[[Y]]#0 (%[[IIDX]])
! CHECK:     %[[YIIP:.*]] = hlfir.designate %[[YI]]{"ip"}
! CHECK:     hlfir.yield %[[YIIP]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xi32>>>>
! CHECK:   }
! CHECK: }

! RHS is an array section. Hits a TODO.
subroutine s5_1(x,y,z,n1,n2)
  use array_of_pointer_test
  type(ta) :: x(:)
  type(tb) :: y(:)
  type(ta), TARGET :: z(:)

  forall (i=1:10)
     ! Slice a rank 1 array and save the slice to the box.
!     x(i)%ip => z(i)%ip(1::n1+1)
  end forall
end subroutine s5_1

! CHECK-LABEL: func.func @_QPs5_1(
! CHECK: hlfir.forall lb {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: } ub {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: }  (%[[IARG:.*]]: i32) {
! CHECK: }

subroutine s6(x,y)
  use array_of_pointer_test
  type(tv) :: x(:)
  integer, target :: y(:)

  forall (i=1:10, j=2:20:2)
     ! Two box indirections.
     x(i)%jp(j)%ip%v = y(i)
  end forall
end subroutine s6

! CHECK-LABEL: func.func @_QPs6(
! CHECK-SAME: %[[XARG:.*]]: !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTtv{{.*}}>>> {{.*}}, %[[YARG:.*]]: !fir.box<!fir.array<?xi32>> {{.*}}) {
! CHECK: %[[X:.*]]:2 = hlfir.declare %[[XARG]]
! CHECK: %[[Y:.*]]:2 = hlfir.declare %[[YARG]]
! CHECK: hlfir.forall lb {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: } ub {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: }  (%[[IARG:.*]]: i32) {
! CHECK:   %[[I:.*]] = hlfir.forall_index "i" %[[IARG]] : (i32) -> !fir.ref<i32>
! CHECK:   hlfir.forall lb {
! CHECK:     hlfir.yield {{.*}} : i32
! CHECK:   } ub {
! CHECK:     hlfir.yield {{.*}} : i32
! CHECK:   } step {
! CHECK:     hlfir.yield {{.*}} : i32
! CHECK:   }  (%[[JARG:.*]]: i32) {
! CHECK:     %[[J:.*]] = hlfir.forall_index "j" %[[JARG]] : (i32) -> !fir.ref<i32>
! CHECK:     hlfir.region_assign {
! CHECK:       %[[IVAL:.*]] = fir.load %[[I]] : !fir.ref<i32>
! CHECK:       %[[IIDX:.*]] = fir.convert %[[IVAL]] : (i32) -> i64
! CHECK:       %[[YI:.*]] = hlfir.designate %[[Y]]#0 (%[[IIDX]])
! CHECK:       %[[YIVAL:.*]] = fir.load %[[YI]] : !fir.ref<i32>
! CHECK:       hlfir.yield %[[YIVAL]] : i32
! CHECK:     } to {
! CHECK:       %[[IVAL:.*]] = fir.load %[[I]] : !fir.ref<i32>
! CHECK:       %[[IIDX:.*]] = fir.convert %[[IVAL]] : (i32) -> i64
! CHECK:       %[[XI:.*]] = hlfir.designate %[[X]]#0 (%[[IIDX]])
! CHECK:       %[[XIJP:.*]] = hlfir.designate %[[XI]]{"jp"}
! CHECK:       %[[JPBOX:.*]] = fir.load %[[XIJP]]
! CHECK:       %[[JVAL:.*]] = fir.load %[[J]] : !fir.ref<i32>
! CHECK:       %[[JIDX:.*]] = fir.convert %[[JVAL]] : (i32) -> i64
! CHECK:       %[[XIJPJ:.*]] = hlfir.designate %[[JPBOX]] (%[[JIDX]])
! CHECK:       %[[XIJPJIP:.*]] = hlfir.designate %[[XIJPJ]]{"ip"}
! CHECK:       %[[IPBOX:.*]] = fir.load %[[XIJPJIP]]
! CHECK:       %[[IPADDR:.*]] = fir.box_addr %[[IPBOX]]
! CHECK:       %[[V:.*]] = hlfir.designate %[[IPADDR]]{"v"}
! CHECK:       hlfir.yield %[[V]] : !fir.ref<i32>
! CHECK:     }
! CHECK:   }
! CHECK: }

subroutine s7(x,y,n)
  use array_of_pointer_test
  type(t) x(:)
  integer, TARGET :: y(:)
  ! Introduce a crossing dependence
  forall (i=1:n)
    x(i)%ip => y(x(n+1-i)%ip)
  end forall
end subroutine s7

! CHECK-LABEL: func.func @_QPs7(
! CHECK-SAME: %[[XARG:.*]]: !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTt{ip:!fir.box<!fir.ptr<i32>>}>>> {{.*}}, %[[YARG:.*]]: !fir.box<!fir.array<?xi32>> {{.*}}, %[[NARG:.*]]: !fir.ref<i32> {{.*}}) {
! CHECK: %[[N:.*]]:2 = hlfir.declare %[[NARG]]
! CHECK: %[[X:.*]]:2 = hlfir.declare %[[XARG]]
! CHECK: %[[Y:.*]]:2 = hlfir.declare %[[YARG]]
! CHECK: hlfir.forall lb {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: } ub {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: }  (%[[IARG:.*]]: i32) {
! CHECK:   %[[I:.*]] = hlfir.forall_index "i" %[[IARG]] : (i32) -> !fir.ref<i32>
! CHECK:   hlfir.region_assign {
! CHECK:     %[[NVAL:.*]] = fir.load %[[N]]#0
! CHECK:     %[[IVAL:.*]] = fir.load %[[I]]
! CHECK:     %[[IDX:.*]] = arith.subi {{.*}}, %[[IVAL]]
! CHECK:     %[[IIDX:.*]] = fir.convert %[[IDX]] : (i32) -> i64
! CHECK:     %[[XNI:.*]] = hlfir.designate %[[X]]#0 (%[[IIDX]])
! CHECK:     %[[XNIIP:.*]] = hlfir.designate %[[XNI]]{"ip"}
! CHECK:     %[[IPBOX:.*]] = fir.load %[[XNIIP]]
! CHECK:     %[[IPADDR:.*]] = fir.box_addr %[[IPBOX]]
! CHECK:     %[[VAL:.*]] = fir.load %[[IPADDR]]
! CHECK:     %[[VAL64:.*]] = fir.convert %[[VAL]] : (i32) -> i64
! CHECK:     %[[YV:.*]] = hlfir.designate %[[Y]]#0 (%[[VAL64]])
! CHECK:     %[[YVBOX:.*]] = fir.embox %[[YV]] : (!fir.ref<i32>) -> !fir.box<!fir.ptr<i32>>
! CHECK:     hlfir.yield %[[YVBOX]] : !fir.box<!fir.ptr<i32>>
! CHECK:   } to {
! CHECK:     %[[IVAL:.*]] = fir.load %[[I]] : !fir.ref<i32>
! CHECK:     %[[IIDX:.*]] = fir.convert %[[IVAL]] : (i32) -> i64
! CHECK:     %[[XI:.*]] = hlfir.designate %[[X]]#0 (%[[IIDX]])
! CHECK:     %[[XIIP:.*]] = hlfir.designate %[[XI]]{"ip"}
! CHECK:     hlfir.yield %[[XIIP]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:   }
! CHECK: }

subroutine s8(x,y,n)
  use array_of_pointer_test
  type(ta) x(:)
  integer, POINTER :: y(:)
  forall (i=1:n)
     x(i)%ip(i:) => y
  end forall
end subroutine s8

! CHECK-LABEL: func.func @_QPs8(
! CHECK-SAME: %[[XARG:.*]]: !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTta{{.*}}>>> {{.*}}, %[[YARG:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {{.*}}, %[[NARG:.*]]: !fir.ref<i32> {{.*}}) {
! CHECK: %[[N:.*]]:2 = hlfir.declare %[[NARG]]
! CHECK: %[[X:.*]]:2 = hlfir.declare %[[XARG]]
! CHECK: %[[Y:.*]]:2 = hlfir.declare %[[YARG]]
! CHECK: hlfir.forall lb {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: } ub {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: }  (%[[IARG:.*]]: i32) {
! CHECK:   %[[I:.*]] = hlfir.forall_index "i" %[[IARG]] : (i32) -> !fir.ref<i32>
! CHECK:   hlfir.region_assign {
! CHECK:     %[[YBOX:.*]] = fir.load %[[Y]]#0
! CHECK:     %[[IVAL:.*]] = fir.load %[[I]] : !fir.ref<i32>
! CHECK:     %[[IIDX:.*]] = fir.convert %[[IVAL]] : (i32) -> i64
! CHECK:     %[[IEXT:.*]] = fir.convert %[[IIDX]] : (i64) -> index
! CHECK:     %[[SHIFT:.*]] = fir.shift %[[IEXT]] : (index) -> !fir.shift<1>
! CHECK:     %[[REBOX:.*]] = fir.rebox %[[YBOX]](%[[SHIFT]]) : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, !fir.shift<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:     hlfir.yield %[[REBOX]] : !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:   } to {
! CHECK:     %[[IVAL:.*]] = fir.load %[[I]] : !fir.ref<i32>
! CHECK:     %[[IIDX:.*]] = fir.convert %[[IVAL]] : (i32) -> i64
! CHECK:     %[[XI:.*]] = hlfir.designate %[[X]]#0 (%[[IIDX]])
! CHECK:     %[[XIIP:.*]] = hlfir.designate %[[XI]]{"ip"}
! CHECK:     hlfir.yield %[[XIIP]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:   }
! CHECK: }

subroutine s8_1(x,y,n1,n2)
  use array_of_pointer_test
  type(ta) x(:)
  integer, POINTER :: y(:)
  forall (i=1:n1)
     x(i)%ip(i:n2+1+i) => y
  end forall
end subroutine s8_1

! CHECK-LABEL: func.func @_QPs8_1(
! CHECK-SAME: %[[XARG:.*]]: !fir.box<!fir.array<?x!fir.type<_QMarray_of_pointer_testTta{{.*}}>>> {{.*}}, %[[YARG:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {{.*}}, %[[N1ARG:.*]]: !fir.ref<i32> {{.*}}, %[[N2ARG:.*]]: !fir.ref<i32> {{.*}}) {
! CHECK: %[[N1:.*]]:2 = hlfir.declare %[[N1ARG]]
! CHECK: %[[N2:.*]]:2 = hlfir.declare %[[N2ARG]]
! CHECK: %[[X:.*]]:2 = hlfir.declare %[[XARG]]
! CHECK: %[[Y:.*]]:2 = hlfir.declare %[[YARG]]
! CHECK: hlfir.forall lb {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: } ub {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: }  (%[[IARG:.*]]: i32) {
! CHECK:   %[[I:.*]] = hlfir.forall_index "i" %[[IARG]] : (i32) -> !fir.ref<i32>
! CHECK:   hlfir.region_assign {
! CHECK:     %[[YBOX:.*]] = fir.load %[[Y]]#0
! CHECK:     %[[IVAL:.*]] = fir.load %[[I]] : !fir.ref<i32>
! CHECK:     %[[IIDX:.*]] = fir.convert %[[IVAL]] : (i32) -> i64
! CHECK:     %[[LB:.*]] = fir.convert %[[IIDX]] : (i64) -> index
! CHECK:     %[[IVAL2:.*]] = fir.load %[[I]]
! CHECK:     %[[UB:.*]] = fir.convert {{.*}} : (i64) -> index
! CHECK:     %[[DIFF:.*]] = arith.subi %[[UB]], %[[LB]]
! CHECK:     %[[DIFFP1:.*]] = arith.addi %[[DIFF]], {{.*}}
! CHECK:     %[[EXTENT:.*]] = arith.select {{.*}}, %[[DIFFP1]], {{.*}}
! CHECK:     %[[SHAPE:.*]] = fir.shape_shift %[[LB]], %[[EXTENT]]
! CHECK:     %[[REBOX:.*]] = fir.rebox %[[YBOX]](%[[SHAPE]])
! CHECK:     hlfir.yield %[[REBOX]] : !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:   } to {
! CHECK:     %[[IVAL:.*]] = fir.load %[[I]] : !fir.ref<i32>
! CHECK:     %[[IIDX:.*]] = fir.convert %[[IVAL]] : (i32) -> i64
! CHECK:     %[[XI:.*]] = hlfir.designate %[[X]]#0 (%[[IIDX]])
! CHECK:     %[[XIIP:.*]] = hlfir.designate %[[XI]]{"ip"}
! CHECK:     hlfir.yield %[[XIIP]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:   }
! CHECK: }

subroutine s8_2(x,y,n)
  use array_of_pointer_test
  type(ta) x(:)
  integer, TARGET :: y(:)
  forall (i=1:n)
!     x(i)%ip(i:) => y
  end forall
end subroutine s8_2

! CHECK-LABEL: func.func @_QPs8_2(
! CHECK: hlfir.forall lb {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: } ub {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: }  (%[[IARG:.*]]: i32) {
! CHECK: }

subroutine s8_3(x,y,n1,n2)
  use array_of_pointer_test
  type(ta) x(:)
  integer, TARGET :: y(:)
  forall (i=1:n1)
!     x(i)%ip(i:n2+1+i) => y
  end forall
end subroutine s8_3

! CHECK-LABEL: func.func @_QPs8_3(
! CHECK: hlfir.forall lb {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: } ub {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: }  (%[[IARG:.*]]: i32) {
! CHECK: }

subroutine s8_4(x,y,n)
  use array_of_pointer_test
  type(ta) x(:)
  integer, ALLOCATABLE, TARGET :: y(:)
  forall (i=1:n)
!     x(i)%ip(i:) => y
  end forall
end subroutine s8_4

! CHECK-LABEL: func.func @_QPs8_4(
! CHECK: hlfir.forall lb {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: } ub {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: }  (%[[IARG:.*]]: i32) {
! CHECK: }

subroutine s8_5(x,y,n1,n2)
  use array_of_pointer_test
  type(ta) x(:)
  integer, ALLOCATABLE, TARGET :: y(:)
  forall (i=1:n1)
!     x(i)%ip(i:n2+1+i) => y
  end forall
end subroutine s8_5

! CHECK-LABEL: func.func @_QPs8_5(
! CHECK: hlfir.forall lb {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: } ub {
! CHECK:   hlfir.yield {{.*}} : i32
! CHECK: }  (%[[IARG:.*]]: i32) {
! CHECK: }
