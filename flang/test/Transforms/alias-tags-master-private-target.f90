! Test case for regression in OpenMP master region with private target integer
! This test was failing with an assertion error in AddAliasTags.cpp
! See issue #172075

! RUN: %flang -fopenmp -c -O1 %s -o %t.o 2>&1 | FileCheck %s --allow-empty

module test
contains
subroutine omp_master_repro()
  implicit none
  integer, parameter :: nim = 4
  integer, parameter :: nvals = 8
  integer, target :: ui
  integer :: hold1(nvals, nim)
  hold1 = 0
  !$OMP PARALLEL DEFAULT(NONE) &
  !$OMP PRIVATE(ui) &
  !$OMP SHARED(hold1, nim)
  !$OMP MASTER
  do ui = 1, nim
     hold1(:, ui) = 1
  end do
  !$OMP END MASTER
  !$OMP END PARALLEL
end subroutine omp_master_repro
end module test
