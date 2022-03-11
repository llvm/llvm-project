! Test internal procedure host association lowering.
! RUN: bbc %s -o - -emit-fir | FileCheck %s

! -----------------------------------------------------------------------------
!     Test non character intrinsic scalars
! -----------------------------------------------------------------------------

!!! Test scalar (with implicit none)

! CHECK-LABEL: func @_QPtest1(
subroutine test1
  implicit none
  integer i
  ! CHECK-DAG: %[[i:.*]] = fir.alloca i32 {{.*}}uniq_name = "_QFtest1Ei"
  ! CHECK-DAG: %[[tup:.*]] = fir.alloca tuple<!fir.ref<i32>>
  ! CHECK: %[[addr:.*]] = fir.coordinate_of %[[tup]], %c0
  ! CHECK: fir.store %[[i]] to %[[addr]] : !fir.llvm_ptr<!fir.ref<i32>>
  ! CHECK: fir.call @_QFtest1Ptest1_internal(%[[tup]]) : (!fir.ref<tuple<!fir.ref<i32>>>) -> ()
  call test1_internal
  print *, i
contains
  ! CHECK-LABEL: func @_QFtest1Ptest1_internal(
  ! CHECK-SAME: %[[arg:[^:]*]]: !fir.ref<tuple<!fir.ref<i32>>> {fir.host_assoc}) {
  ! CHECK: %[[iaddr:.*]] = fir.coordinate_of %[[arg]], %c0
  ! CHECK: %[[i:.*]] = fir.load %[[iaddr]] : !fir.llvm_ptr<!fir.ref<i32>>
  ! CHECK: %[[val:.*]] = fir.call @_QPifoo() : () -> i32
  ! CHECK: fir.store %[[val]] to %[[i]] : !fir.ref<i32>
  subroutine test1_internal
    integer, external :: ifoo
    i = ifoo()
  end subroutine test1_internal
end subroutine test1

!!! Test scalar

! CHECK-LABEL: func @_QPtest2() {
subroutine test2
  a = 1.0
  b = 2.0
  ! CHECK: %[[tup:.*]] = fir.alloca tuple<!fir.ref<f32>, !fir.ref<f32>>
  ! CHECK: %[[a0:.*]] = fir.coordinate_of %[[tup]], %c0
  ! CHECK: fir.store %{{.*}} to %[[a0]] : !fir.llvm_ptr<!fir.ref<f32>>
  ! CHECK: %[[b0:.*]] = fir.coordinate_of %[[tup]], %c1
  ! CHECK: fir.store %{{.*}} to %[[b0]] : !fir.llvm_ptr<!fir.ref<f32>>
  ! CHECK: fir.call @_QFtest2Ptest2_internal(%[[tup]]) : (!fir.ref<tuple<!fir.ref<f32>, !fir.ref<f32>>>) -> ()
  call test2_internal
  print *, a, b
contains
  ! CHECK-LABEL: func @_QFtest2Ptest2_internal(
  ! CHECK-SAME: %[[arg:[^:]*]]: !fir.ref<tuple<!fir.ref<f32>, !fir.ref<f32>>> {fir.host_assoc}) {
  subroutine test2_internal
    ! CHECK: %[[a:.*]] = fir.coordinate_of %[[arg]], %c0
    ! CHECK: %[[aa:.*]] = fir.load %[[a]] : !fir.llvm_ptr<!fir.ref<f32>>
    ! CHECK: %[[b:.*]] = fir.coordinate_of %[[arg]], %c1
    ! CHECK: %{{.*}} = fir.load %[[b]] : !fir.llvm_ptr<!fir.ref<f32>>
    ! CHECK: fir.alloca
    ! CHECK: fir.load %[[aa]] : !fir.ref<f32>
    c = a
    a = b
    b = c
    call test2_inner
  end subroutine test2_internal

  ! CHECK-LABEL: func @_QFtest2Ptest2_inner(
  ! CHECK-SAME: %[[arg:[^:]*]]: !fir.ref<tuple<!fir.ref<f32>, !fir.ref<f32>>> {fir.host_assoc}) {
  subroutine test2_inner
    ! CHECK: %[[a:.*]] = fir.coordinate_of %[[arg]], %c0
    ! CHECK: %[[aa:.*]] = fir.load %[[a]] : !fir.llvm_ptr<!fir.ref<f32>>
    ! CHECK: %[[b:.*]] = fir.coordinate_of %[[arg]], %c1
    ! CHECK: %[[bb:.*]] = fir.load %[[b]] : !fir.llvm_ptr<!fir.ref<f32>>
    ! CHECK-DAG: %[[bd:.*]] = fir.load %[[bb]] : !fir.ref<f32>
    ! CHECK-DAG: %[[ad:.*]] = fir.load %[[aa]] : !fir.ref<f32>
    ! CHECK: %{{.*}} = arith.cmpf ogt, %[[ad]], %[[bd]] : f32
    if (a > b) then
       b = b + 2.0
    end if
  end subroutine test2_inner
end subroutine test2

! -----------------------------------------------------------------------------
!     Test non character scalars
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QPtest6(
! CHECK-SAME: %[[c:.*]]: !fir.boxchar<1>
subroutine test6(c)
  character(*) :: c
  ! CHECK: %[[cunbox:.*]]:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[tup:.*]] = fir.alloca tuple<!fir.boxchar<1>>
  ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[tup]], %c0{{.*}} : (!fir.ref<tuple<!fir.boxchar<1>>>, i32) -> !fir.ref<!fir.boxchar<1>>
  ! CHECK: %[[emboxchar:.*]] = fir.emboxchar %[[cunbox]]#0, %[[cunbox]]#1 : (!fir.ref<!fir.char<1,?>>, index) -> !fir.boxchar<1>
  ! CHECK: fir.store %[[emboxchar]] to %[[coor]] : !fir.ref<!fir.boxchar<1>>
  ! CHECK: fir.call @_QFtest6Ptest6_inner(%[[tup]]) : (!fir.ref<tuple<!fir.boxchar<1>>>) -> ()
  call test6_inner
  print *, c

contains
  ! CHECK-LABEL: func @_QFtest6Ptest6_inner(
  ! CHECK-SAME: %[[tup:.*]]: !fir.ref<tuple<!fir.boxchar<1>>> {fir.host_assoc}) {
  subroutine test6_inner
    ! CHECK: %[[coor:.*]] = fir.coordinate_of %[[tup]], %c0{{.*}} : (!fir.ref<tuple<!fir.boxchar<1>>>, i32) -> !fir.ref<!fir.boxchar<1>>
    ! CHECK: %[[load:.*]] = fir.load %[[coor]] : !fir.ref<!fir.boxchar<1>>
    ! CHECK: fir.unboxchar %[[load]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    c = "Hi there"
  end subroutine test6_inner
end subroutine test6
