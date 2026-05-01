! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -Wno-openmp-threadprivate-equivalence
! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -Wno-open-mp-threadprivate-equivalence

! This is a test for an extension to the OpenMP semantics, see https://github.com/llvm/llvm-project/issues/180493

program threadprivate02
  common /blk1/ a1
  real :: a1
  real :: eq_a
  equivalence(eq_a, a1)

  !$omp threadprivate(/blk1/)

  !ERROR: A THREADPRIVATE variable cannot be in SHARED clause
  !$omp parallel shared(eq_a)
  !$omp end parallel
end
