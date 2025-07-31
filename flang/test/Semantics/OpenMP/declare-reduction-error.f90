! RUN: not %flang_fc1 -emit-obj -fopenmp -fopenmp-version=50 %s 2>&1 | FileCheck %s

subroutine initme(x,n)
  integer x,n
  x=n
end subroutine initme

subroutine subr
  !$omp declare reduction(red_add:integer(4):omp_out=omp_out+omp_in) initializer(initme(omp_priv,0))
  !CHECK: error: Implicit subroutine declaration 'initme' in DECLARE REDUCTION
end subroutine subr
