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

subroutine same_base_variant
!ERROR: The variant procedure must differ from the base procedure
  !$omp declare variant (sub:sub) match (construct={parallel})
contains
  subroutine sub
  end subroutine
end subroutine

subroutine duplicate_variant
  !$omp declare variant (sub:vsub) match (construct={parallel})
!ERROR: Variant 'vsub' was already specified for 'sub' in another DECLARE VARIANT directive
  !$omp declare variant (sub:vsub) match (construct={teams})
contains
  subroutine vsub
  end subroutine
  subroutine sub
  end subroutine
end subroutine

subroutine invalid_clause
!ERROR: PRIVATE clause is not allowed on the DECLARE VARIANT directive
  !$omp declare variant (sub:vsub) match (construct={parallel}) private(x)
contains
  subroutine vsub
  end subroutine
  subroutine sub
    integer :: x
  end subroutine
end subroutine
