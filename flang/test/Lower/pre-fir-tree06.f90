! RUN: %flang_fc1 -fdebug-pre-fir-tree -fopenmp %s | FileCheck %s

! Test structure of the Pre-FIR tree with OpenMP declarative construct

! CHECK: ModuleLike
module m
  real, dimension(10) :: x
  ! CHECK-NEXT: OpenMPDeclarativeConstruct
  !$omp threadprivate(x)
end
! CHECK: End ModuleLike

