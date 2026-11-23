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
  integer :: a
!ERROR: A variable that appears in a DECLARE TARGET directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
  !$omp declare target(bar)
  a = bar
!ERROR: 'bar' is not a callable procedure
  a = bar(a)
end subroutine

subroutine f05
  integer :: a
!ERROR: A variable that appears in a DECLARE TARGET directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
  !$omp declare target link(bar)
!ERROR: 'bar' is not a callable procedure
  a = bar(a)
end subroutine

subroutine f06
!ERROR: The entity with PARAMETER attribute is used in a DECLARE TARGET directive [-Wopenmp-usage]
  !$omp declare target(bar)
  parameter (bar = 1)
end

subroutine f07
  !$omp declare target(bar)
  save :: bar
  integer :: a
!ERROR: 'bar' is not a callable procedure
  a = bar(a)
end

subroutine f08
!ERROR: A variable that appears in a DECLARE TARGET directive must be declared in the scope of a module or have the SAVE attribute, either explicitly or implicitly
  !$omp declare target(bar)
  dimension bar(1:10)
  integer :: a
  a = bar(a)
end

module f09
  !$omp declare target(baz)
  codimension :: baz[*]
contains
subroutine f10
!ERROR: Cannot call function 'baz' like a subroutine
  call baz()
end
end module
