! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 -o - %s | FileCheck %s

subroutine func7(a)
  implicit none
  double precision :: a(100, 100)
  integer :: i, j

  !$omp parallel do
  !$omp tile sizes(4, 16)
  do i = 1, 100
    do j = 1, 100
      a(j, i) = a(j, i) + 1.0d0
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPfunc7(
! CHECK: omp.parallel
! CHECK: omp.wsloop
! CHECK: omp.loop_nest
! CHECK-SAME: tiles(4, 16)

