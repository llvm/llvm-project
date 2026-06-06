!RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp

subroutine baz(x)
  integer :: x
end

subroutine f
!ERROR: A variable in a DECLARE TARGET directive cannot be an element of a common block
  !$omp declare target(baz)
  integer :: p
  common /xx/ baz, p
!ERROR: Cannot call function 'baz' like a subroutine
  call baz(p)
end

subroutine g
!ERROR: A variable in a DECLARE TARGET directive cannot be an element of a common block
  !$omp declare target(baz)
  integer :: p, q
  common /yy/ baz, p
!ERROR: 'baz' is not a callable procedure
  q = baz(p)
end
