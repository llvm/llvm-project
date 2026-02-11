!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 %s -o - | FileCheck %s

! Test lowering of groupprivate directive to omp.groupprivate.

! CHECK-DAG: fir.global common @blk_
! CHECK-DAG: fir.global @_QMmEx : i32

! Test: basic groupprivate with single variable

module m
  implicit none
  integer, save :: x
  !$omp groupprivate(x)
end module

! CHECK-LABEL: func.func @_QPtest_groupprivate
! CHECK: omp.target
! CHECK:   omp.teams
! CHECK:     fir.address_of(@_QMmEx)
! CHECK:     omp.groupprivate
subroutine test_groupprivate()
  use m

  !$omp target
    !$omp teams
      x = 10
    !$omp end teams
  !$omp end target
end subroutine

! Test: groupprivate with common block
module m2
  implicit none
  integer :: cb_x, cb_y
  real :: cb_z
  common /blk/ cb_x, cb_y, cb_z
  !$omp groupprivate(/blk/)
end module

! CHECK-LABEL: func.func @_QPtest_common_block_groupprivate
! CHECK: omp.target
! CHECK:   omp.teams
! CHECK:     fir.address_of(@blk_)
! CHECK:     omp.groupprivate
! CHECK:     fir.convert
! CHECK:     fir.coordinate_of
subroutine test_common_block_groupprivate()
  use m2

  !$omp target
    !$omp teams
      cb_x = 1
      cb_y = 2
      cb_z = 3.0
    !$omp end teams
  !$omp end target
end subroutine
