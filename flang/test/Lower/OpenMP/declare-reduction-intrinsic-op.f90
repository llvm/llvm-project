! RUN: not %flang_fc1 -emit-mlir -fopenmp %s -o - 2>&1 | FileCheck %s

program test
  type t
     integer :: x
  end type t
  ! CHECK: not yet implemented: Reduction of some types is not supported
  !$omp declare reduction(+:t: omp_out%x = omp_out%x + omp_in%x) initializer(omp_priv = t(0))
  type(t) :: a
  a = t(0)
  !$omp parallel reduction(+:a)
  a%x = a%x + 1
  !$omp end parallel
end program test
