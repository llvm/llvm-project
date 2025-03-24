!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

!RUN: %flang_fc1 -emit-hlfir -fopenmp -O3 %s -o - | FileCheck %s --check-prefix HLFIR
!RUN: %flang_fc1 -emit-fir -fopenmp -O3 %s -o - | FileCheck %s --check-prefix FIR

subroutine sb1(x, y)
  integer :: x(:)
  integer :: y(:)
  !$omp parallel workshare
  x = y
  !$omp end parallel workshare
end subroutine

! HLFIR:     omp.parallel {
! HLFIR:       omp.workshare {
! HLFIR:         hlfir.assign
! HLFIR:         omp.terminator
! HLFIR:       }
! HLFIR:       omp.terminator
! HLFIR:     }

! FIR:     omp.parallel {
! FIR:       omp.wsloop nowait {
! FIR:         omp.loop_nest
! FIR:       }
! FIR:       omp.barrier
! FIR:       omp.terminator
! FIR:     }
