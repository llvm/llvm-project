! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52

! MATCH clause checks for DECLARE VARIANT: required/duplicate clause and
! context-selector validation (shared with METADIRECTIVE).

subroutine f00
  !$omp declare variant (sub:vsub) &
  !$omp & match (implementation={vendor("this")}, &
!ERROR: Repeated trait set name IMPLEMENTATION in a context specifier
  !$omp &       implementation={requires(unified_shared_memory)})
contains
  subroutine vsub
  end subroutine
  subroutine sub
  end subroutine
end subroutine

subroutine f01
  !$omp declare variant (sub:vsub) &
!ERROR: Repeated trait name ISA in a trait set
  !$omp & match (device={isa("this"), isa("that")})
contains
  subroutine vsub
  end subroutine
  subroutine sub
  end subroutine
end subroutine

subroutine f02
  !$omp declare variant (sub:vsub) &
!ERROR: SCORE expression must be a non-negative constant integer expression
  !$omp & match (user={condition(score(-2): .true.)})
contains
  subroutine vsub
  end subroutine
  subroutine sub
  end subroutine
end subroutine

subroutine f03(x)
  integer :: x
  !$omp declare variant (sub:vsub) &
!ERROR: SCORE expression must be a non-negative constant integer expression
  !$omp & match (user={condition(score(x): .true.)})
contains
  subroutine vsub
  end subroutine
  subroutine sub
  end subroutine
end subroutine

subroutine f04
  !$omp declare variant (sub:vsub) &
!ERROR: Trait property should be a scalar expression
!ERROR: More invalid properties are present
  !$omp & match (target_device={device_num("device", "foo"(1))})
contains
  subroutine vsub
  end subroutine
  subroutine sub
  end subroutine
end subroutine

subroutine f05(x)
  integer :: x
  !$omp declare variant (sub:vsub) &
  !$omp & match (user={ &
!ERROR: CONDITION trait requires a single LOGICAL expression
  !$omp & condition(score(2): x)})
contains
  subroutine vsub
  end subroutine
  subroutine sub
  end subroutine
end subroutine

subroutine f06(x)
  integer :: x
!ERROR: Run-time USER condition in the MATCH clause is not yet implemented
  !$omp declare variant (sub:vsub) match (user={condition(x > 0)})
contains
  subroutine vsub
  end subroutine
  subroutine sub
  end subroutine
end subroutine

subroutine f07
  !$omp declare variant (sub:vsub) &
!ERROR: SCORE is not allowed for DEVICE trait set
  !$omp & match (device={kind(score(1): host)})
contains
  subroutine vsub
  end subroutine
  subroutine sub
  end subroutine
end subroutine

subroutine f08
!ERROR: DECLARE_VARIANT directive requires a MATCH clause
  !$omp declare variant (sub:vsub)
contains
  subroutine vsub
  end subroutine
  subroutine sub
  end subroutine
end subroutine

subroutine f09
  !$omp declare variant (sub:vsub) match (construct={parallel}) &
!ERROR: At most one MATCH clause can appear on DECLARE VARIANT directive
  !$omp & match (construct={teams})
contains
  subroutine vsub
  end subroutine
  subroutine sub
  end subroutine
end subroutine

! ADJUST_ARGS is accepted by parsing but variant selection does not honour it
! yet.
subroutine f10
contains
  subroutine vsub(v1)
    integer, value :: v1
  end subroutine
  subroutine sub(v1)
    integer, value :: v1
!ERROR: ADJUST_ARGS clause on the DECLARE VARIANT directive is not yet implemented
    !$omp declare variant(vsub) match(construct={dispatch}) adjust_args(nothing: v1)
  end subroutine
end subroutine

! APPEND_ARGS is likewise not yet honoured.
subroutine f11
contains
  subroutine vsub(v1)
    integer, value :: v1
  end subroutine
  subroutine sub(v1)
    integer, value :: v1
!ERROR: APPEND_ARGS clause on the DECLARE VARIANT directive is not yet implemented
    !$omp declare variant(vsub) match(construct={dispatch}) append_args(interop(target))
  end subroutine
end subroutine
