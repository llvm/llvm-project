! Regression test for crash compiling privatizer for a pointer to an array.
! The crash was because the fir.embox was not given a shape but it needs one.

!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! ALLOCATABLE case (2nd subroutine)
!CHECK-LABEL: omp.private {type = firstprivate}
!CHECK-SAME: @{{.*}} : !fir.box<!fir.heap<!fir.array<?x!fir.type<{{.*}}>>>> init {
!CHECK:        if %{{.*}} {
!CHECK:        %[[SHAPE:.*]] = fir.shape
!CHECK:        %[[BOX:.*]] = fir.embox %{{.*}}(%[[SHAPE]])
!CHECK:        } else {

! POINTER case (1st subroutine)
!CHECK-LABEL: omp.private {type = firstprivate}
!CHECK-SAME: @{{.*}} : !fir.box<!fir.ptr<!fir.array<?x!fir.type<{{.*}}>>>> init {
!CHECK:        %[[SHAPE:.*]] = fir.shape
!CHECK:        %[[ADDR:.*]] = fir.zero_bits
!CHECK:        %[[BOX:.*]] = fir.embox %[[ADDR]](%[[SHAPE]])

subroutine pointer_to_array_derived
  type t
    integer :: i
  end type
  type(t), pointer :: a(:)
  allocate(a(1))
  a(1)%i = 2
  !$omp parallel firstprivate(a)
  if (a(1)%i/=2) stop 2
  !$omp end parallel
end subroutine

subroutine allocatable_array_derived
  type t
    integer :: i
  end type
  type(t), allocatable :: a(:)
  allocate(a(1))
  a(1)%i = 2
  !$omp parallel firstprivate(a)
  if (a(1)%i/=2) stop 2
  !$omp end parallel
end subroutine
