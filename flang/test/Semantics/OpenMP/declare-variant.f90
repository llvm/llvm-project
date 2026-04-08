! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51

subroutine sub0
!ERROR: The name 'vsub1' should refer to a procedure
  !$omp declare variant (sub:vsub1) match (construct={parallel})
!ERROR: The name 'sub1' should refer to a procedure
  !$omp declare variant (sub1:vsub) match (construct={parallel})
contains
  subroutine vsub
  end subroutine

  subroutine sub ()
  end subroutine
end subroutine
