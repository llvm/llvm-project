!RUN: %flang_fc1 -fopenmp -emit-fir -o - %s | FileCheck %s
!RUN: bbc -fopenmp -emit-fir -o - %s | FileCheck %s

!Allow POINTER variables in OpenMP SHARED clause. Check that this
!code compiles.

!CHECK-LABEL: func.func @_QPfoo
subroutine foo()
  procedure(), pointer :: pf
  !$omp parallel shared(pf)
  !$omp end parallel
end

