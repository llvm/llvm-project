! This test checks lowering of OpenMP ordered directive with threads Clause.
! Without clause in ordered direcitve, it behaves as if threads clause is
! specified.

!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

subroutine ordered
        integer :: i
        integer :: a(20)

!CHECK: omp.ordered_region  {
!$OMP ORDERED
        a(i) = a(i-1) + 1
!CHECK:   omp.terminator
!CHECK-NEXT: }
!$OMP END ORDERED

!CHECK: omp.ordered_region  {
!$OMP ORDERED THREADS
        a(i) = a(i-1) + 1
!CHECK:   omp.terminator
!CHECK-NEXT: }
!$OMP END ORDERED

end
