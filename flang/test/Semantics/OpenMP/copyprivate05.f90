!RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp

! The first testcase from https://github.com/llvm/llvm-project/issues/141481

subroutine f00
  type t
  end type

!ERROR: 't' must be a variable
!$omp single copyprivate(t)
!$omp end single
end
