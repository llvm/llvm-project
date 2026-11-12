! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp

! This is a test for an extension to the OpenMP semantics, see https://github.com/llvm/llvm-project/issues/180493

program threadprivate02
  common /blk1/ a1
  real :: a1
  real :: eq_a
  equivalence(eq_a, a1)

  !WARNING: A variable in a THREADPRIVATE directive used in an EQUIVALENCE statement is an OpenMP extension (variable 'a1' from common block '/blk1/') [-Wopenmp-threadprivate-equivalence]
  !$omp threadprivate(/blk1/)

  !ERROR: A THREADPRIVATE variable cannot be in SHARED clause
  !$omp parallel shared(eq_a)
  !$omp end parallel
end
