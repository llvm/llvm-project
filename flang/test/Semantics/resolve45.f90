! RUN: %python %S/test_errors.py %s %flang_fc1
!ERROR: The function result variable 'f1' may not have an explicit SAVE attribute
function f1(x, y)
  !ERROR: The dummy argument 'x' may not have an explicit SAVE attribute
  integer x
  save x,y
  !ERROR: The dummy argument 'y' may not have an explicit SAVE attribute
  integer y
  save f1
end

!ERROR: The entity 'f2' with an explicit SAVE attribute must be a variable, procedure pointer, or COMMON block
function f2(x, y) result(r)
  save f2
  !ERROR: The function result variable 'r' may not have an explicit SAVE attribute
  real, save :: r
  !ERROR: The dummy argument 'x' may not have an explicit SAVE attribute
  complex, save :: x
  allocatable :: y
  !ERROR: The dummy argument 'y' may not have an explicit SAVE attribute
  integer :: y
  save :: y
end

! SAVE statement should not trigger the above errors
function f2b(x, y)
  real :: x, y
  save
end

subroutine s3(x)
  !ERROR: The dummy argument 'x' may not have an explicit SAVE attribute
  procedure(integer), pointer, save :: x
  !ERROR: The entity 'y' with an explicit SAVE attribute must be a variable, procedure pointer, or COMMON block
  procedure(integer), save :: y
end

subroutine s4
  !WARNING: Explicit SAVE of 'z' is redundant due to global SAVE statement
  save z
  save
  procedure(integer), pointer :: x
  !WARNING: Explicit SAVE of 'x' is redundant due to global SAVE statement
  save :: x
  !WARNING: Explicit SAVE of 'y' is redundant due to global SAVE statement
  integer, save :: y
end

subroutine s5
  implicit none
  integer x
  block
    !ERROR: No explicit type declared for 'x'
    save x
  end block
end

subroutine s7
  !ERROR: 'x' appears as a COMMON block in a SAVE statement but not in a COMMON statement
  save /x/
end

subroutine s8a(n)
  integer :: n
  real :: x(n)  ! OK: save statement doesn't affect x
  save
end
subroutine s8b(n)
  integer :: n
  !ERROR: The automatic object 'x' may not have an explicit SAVE attribute
  real, save :: x(n)
end
