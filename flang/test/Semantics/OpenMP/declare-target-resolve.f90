!RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp

function bar(x)
  integer :: bar, x
end

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
  !$omp declare target(bar)
  integer :: p, q
  equivalence(bar, p)
!ERROR: 'bar' is not a callable procedure
  q = bar(p)
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
  !$omp declare target(bar)
  integer :: p, q
  common /yy/ bar, p
!ERROR: 'bar' is not a callable procedure
  q = bar(p)
end

subroutine f04
  real :: a
!ERROR: A variable that appears in a DECLARE TARGET directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
  !$omp declare target(bar)
  a = bar
!ERROR: 'bar' is not a callable procedure
  a = bar(a)
end subroutine

subroutine f05
  real :: a
!ERROR: A variable that appears in a DECLARE TARGET directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
  !$omp declare target link(bar)
!ERROR: 'bar' is not a callable procedure
  a = bar(a)
end subroutine

