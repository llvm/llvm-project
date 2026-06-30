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
!ERROR: PRIVATE clause is not allowed on DECLARE VARIANT directive
  !$omp declare variant (sub:vsub) match (construct={parallel}) private(x)
contains
  subroutine vsub
  end subroutine
  subroutine sub
    integer :: x
  end subroutine
end subroutine

subroutine incompatible_argcount
!ERROR: The variant procedure 'vsub' is not compatible with the base procedure 'sub': distinct numbers of dummy arguments
  !$omp declare variant (sub:vsub) match (construct={parallel})
contains
  subroutine sub(x)
    integer :: x
  end subroutine
  subroutine vsub(x, y)
    integer :: x, y
  end subroutine
end subroutine

subroutine incompatible_argtype
!ERROR: The variant procedure 'vsub' is not compatible with the base procedure 'sub': incompatible dummy argument #1: incompatible dummy data object types: REAL(4) vs INTEGER(4)
  !$omp declare variant (sub:vsub) match (construct={parallel})
contains
  subroutine sub(x)
    integer :: x
  end subroutine
  subroutine vsub(x)
    real :: x
  end subroutine
end subroutine

subroutine incompatible_function_vs_subroutine
!ERROR: The variant procedure 'vfun' is not compatible with the base procedure 'sub': incompatible procedures: one is a function, the other a subroutine
  !$omp declare variant (sub:vfun) match (construct={parallel})
contains
  subroutine sub(x)
    integer :: x
  end subroutine
  integer function vfun(x)
    integer :: x
    vfun = x
  end function
end subroutine

subroutine incompatible_result
!ERROR: The variant procedure 'fvar' is not compatible with the base procedure 'fbase': function results have distinct types: INTEGER(4) vs REAL(4)
  !$omp declare variant (fbase:fvar) match (construct={parallel})
contains
  integer function fbase(x)
    integer :: x
    fbase = x
  end function
  real function fvar(x)
    real :: x
    fvar = x
  end function
end subroutine

! When the base name is omitted, the enclosing procedure is the base

subroutine incompatible_omitted_base(x)
  integer :: x
!ERROR: The variant procedure 'vsub' is not compatible with the base procedure 'incompatible_omitted_base': distinct numbers of dummy arguments
  !$omp declare variant (vsub) match (construct={parallel})
contains
  subroutine vsub(x, y)
    integer :: x, y
  end subroutine
end subroutine

! Differing dummy argument names are fine; only characteristics matter.

subroutine compatible_interface
  !$omp declare variant (sub:vsub) match (construct={parallel})
contains
  subroutine sub(x)
    integer :: x
  end subroutine
  subroutine vsub(y)
    integer :: y
  end subroutine
end subroutine

! append_args is rejected as not-yet-implemented. The interface check is also
! skipped (the appended interop argument intentionally changes the variant
! interface), so the mismatched argument count does not produce a spurious
! incompatibility error in addition to the not-yet-implemented one.

subroutine append_args_skips_interface_check
!ERROR: APPEND_ARGS clause on the DECLARE VARIANT directive is not yet implemented
  !$omp declare variant (sub:vsub) match (construct={dispatch}) append_args(interop(target))
contains
  subroutine sub(x)
    integer :: x
  end subroutine
  subroutine vsub(x, obj)
    integer :: x, obj
  end subroutine
end subroutine
