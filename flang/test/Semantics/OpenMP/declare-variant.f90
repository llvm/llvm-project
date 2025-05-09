! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51

subroutine sub0
!ERROR: Implicit subroutine declaration 'vsub1' in !$OMP DECLARE VARIANT
  !$omp declare variant (sub:vsub1) match (construct={parallel})
!ERROR: Implicit subroutine declaration 'sub1' in !$OMP DECLARE VARIANT
  !$omp declare variant (sub1:vsub) match (construct={parallel})
contains
  subroutine vsub
  end subroutine

  subroutine sub ()
  end subroutine
end subroutine
