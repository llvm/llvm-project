!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s
!RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPfoo
subroutine foo
  ! CHECK-NOT: omp.
  !$omp nothing
end subroutine
