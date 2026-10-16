! RUN: %flang_fc1 -fopenmp -emit-hlfir %s -o - | FileCheck %s

program p
  type y3; integer, allocatable :: x; end type
  type(y3) :: v
  integer :: s, n, i
  s = 1; n = 10
  allocate(v%x); v%x = 0
!$omp parallel
  if (.not. allocated(v%x)) print *, '101', allocated(v%x)
!$omp do schedule(dynamic) lastprivate(v)
  do i = s, n
    v%x = i
  end do
!$omp end do
!$omp end parallel
end program

! CHECK:      omp.parallel {
! CHECK-NOT:  private(
! CHECK:      omp.wsloop
! CHECK-SAME: private(
