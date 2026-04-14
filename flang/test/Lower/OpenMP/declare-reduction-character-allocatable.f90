! Test OpenMP declare reduction with character type and allocatable variable.
! This test ensures that Flang doesn't crash when processing user-defined
! reductions with character types on allocatable variables.
! See issue: https://github.com/llvm/llvm-project/issues/177501

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! Test basic character reduction with allocatable variable (constant length)
program test_character_reduction
  character(len=1), allocatable :: k1

  !$omp declare reduction (char_max:character(len=1):omp_out=max(omp_out,omp_in)) &
  !$omp initializer(omp_priv='a')

  !$omp parallel sections reduction(char_max:k1)
    k1 = max(k1, 'z')
  !$omp end parallel sections

end program test_character_reduction

! Verify the declare_reduction is generated with reference type for character
! CHECK-LABEL: omp.declare_reduction @char_max : !fir.ref<!fir.char<1>>
! CHECK: init {
! CHECK: omp.yield

! Verify the combiner region works
! CHECK: combiner
! CHECK: hlfir.declare
! CHECK: omp.yield

! Verify the reduction is used in the parallel sections
! CHECK: omp.parallel
! CHECK:   omp.sections reduction(byref @char_max
