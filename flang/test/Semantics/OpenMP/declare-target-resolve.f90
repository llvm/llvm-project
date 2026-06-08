!RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp

subroutine baz(x)
  integer :: x
end

subroutine f00
!ERROR: A variable in a DECLARE TARGET directive cannot appear in an EQUIVALENCE statement
  !$omp declare target(baz)
  integer :: p
  equivalence(baz, p)
!ERROR: Cannot call function 'baz' like a subroutine
  call baz(p)
end

subroutine f01
!ERROR: A variable in a DECLARE TARGET directive cannot appear in an EQUIVALENCE statement
  !$omp declare target(baz)
  integer :: p, q
  equivalence(baz, p)
!ERROR: 'baz' is not a callable procedure
  q = baz(p)
end

subroutine f02
!ERROR: A variable in a DECLARE TARGET directive cannot be an element of a common block
  !$omp declare target(baz)
  integer :: p
  common /xx/ baz, p
!ERROR: Cannot call function 'baz' like a subroutine
  call baz(p)
end

subroutine f03
!ERROR: A variable in a DECLARE TARGET directive cannot be an element of a common block
  !$omp declare target(baz)
  integer :: p, q
  common /yy/ baz, p
!ERROR: 'baz' is not a callable procedure
  q = baz(p)
end

subroutine f04
  real :: a
!ERROR: A variable that appears in a DECLARE TARGET directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
  !$omp declare target(baz)
  a = baz
!ERROR: 'baz' is not a callable procedure
  a = baz(a)
end subroutine

