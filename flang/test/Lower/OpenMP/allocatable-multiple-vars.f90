! Test early privatization for multiple allocatable variables.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --openmp-enable-delayed-privatization=false \
! RUN:   -o - %s 2>&1 | FileCheck %s

! RUN: bbc -emit-hlfir -fopenmp --openmp-enable-delayed-privatization=false -o - %s 2>&1 |\
! RUN:   FileCheck %s

subroutine delayed_privatization_allocatable
  implicit none
  integer, allocatable :: var1, var2

!$omp parallel private(var1, var2)
  var1 = 10
  var2 = 20
!$omp end parallel
end subroutine

! Verify that private versions of each variable are both allocated and freed
! within the parallel region.

! CHECK:      omp.parallel {
! CHECK:        fir.allocmem
! CHECK:        fir.allocmem
! CHECK:        fir.freemem
! CHECK:        fir.freemem
! CHECK:        omp.terminator
! CHECK-NEXT: }
