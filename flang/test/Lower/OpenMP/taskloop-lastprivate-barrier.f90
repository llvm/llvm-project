! TASKLOOP must not request a privatization barrier; worksharing loops still do.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

! Allocatable privatizer reads the original (mold).
! CHECK-LABEL: func.func @_QPtaskloop_lastprivate_allocatable
! CHECK:         omp.taskloop.context private(@{{.*}}Ea_private_box_heap_Uxi32
! CHECK-NOT:     private_barrier
! CHECK-SAME:    {
! CHECK-NOT:     omp.barrier
! CHECK:         omp.taskloop.wrapper
subroutine taskloop_lastprivate_allocatable()
  integer, allocatable :: a(:)
  integer :: i
  allocate(a(100))
  a = -1
  !$omp taskloop lastprivate(a)
  do i = 1, 1
    a(i) = i
  end do
  !$omp end taskloop
end subroutine

! Same variable firstprivate and lastprivate.
! CHECK-LABEL: func.func @_QPtaskloop_first_and_lastprivate
! CHECK:         omp.taskloop.context private(@{{.*}}Ex_firstprivate_i32
! CHECK-NOT:     private_barrier
! CHECK-SAME:    {
! CHECK-NOT:     omp.barrier
! CHECK:         omp.taskloop.wrapper
subroutine taskloop_first_and_lastprivate()
  integer :: x, i
  x = 5
  !$omp taskloop firstprivate(x) lastprivate(x)
  do i = 1, 4
    x = x + i
  end do
  !$omp end taskloop
end subroutine

! CHECK-LABEL: func.func @_QPtaskloop_nogroup_first_and_lastprivate
! CHECK:         omp.taskloop.context nogroup private(@{{.*}}Ex_firstprivate_i32
! CHECK-NOT:     private_barrier
! CHECK-SAME:    {
subroutine taskloop_nogroup_first_and_lastprivate()
  integer :: x, i
  x = 5
  !$omp taskloop nogroup firstprivate(x) lastprivate(x)
  do i = 1, 4
    x = x + i
  end do
  !$omp end taskloop
end subroutine

! DO keeps the barrier.
! CHECK-LABEL: func.func @_QPdo_first_and_lastprivate
! CHECK:         omp.wsloop private(@{{.*}}Ex_firstprivate_i32 {{.*}}) private_barrier {
subroutine do_first_and_lastprivate()
  integer :: x, i
  x = 5
  !$omp parallel
  !$omp do firstprivate(x) lastprivate(x)
  do i = 1, 4
    x = x + i
  end do
  !$omp end do
  !$omp end parallel
end subroutine
