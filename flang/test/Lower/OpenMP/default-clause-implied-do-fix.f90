!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

!CHECK: @_QPsb
subroutine sb(a)
  integer :: a(:)
!CHECK: omp.parallel
  !$omp parallel default(private)
!CHECK: hlfir.elemental
    if (any(a/=(/(100,i=1,5)/))) print *, "OK"
  !$omp end parallel
end subroutine
