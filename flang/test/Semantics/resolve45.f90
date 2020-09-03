! RUN: %S/test_errors.sh %s %t %f18
function f1(x, y)
  integer x
  !ERROR: SAVE attribute may not be applied to dummy argument 'x'
  !ERROR: SAVE attribute may not be applied to dummy argument 'y'
  save x,y
  integer y
  !ERROR: SAVE attribute may not be applied to function result 'f1'
  save f1
end

function f2(x, y)
  !ERROR: SAVE attribute may not be applied to function result 'f2'
  real, save :: f2
  !ERROR: SAVE attribute may not be applied to dummy argument 'x'
  complex, save :: x
  allocatable :: y
  integer :: y
  !ERROR: SAVE attribute may not be applied to dummy argument 'y'
  save :: y
end

! SAVE statement should not trigger the above errors
function f2b(x, y)
  real :: x, y
  save
end

subroutine s3(x)
  !ERROR: SAVE attribute may not be applied to dummy argument 'x'
  procedure(integer), pointer, save :: x
  !ERROR: Procedure 'y' with SAVE attribute must also have POINTER attribute
  procedure(integer), save :: y
end

subroutine s4
  !ERROR: Explicit SAVE of 'z' is redundant due to global SAVE statement
  save z
  save
  procedure(integer), pointer :: x
  !ERROR: Explicit SAVE of 'x' is redundant due to global SAVE statement
  save :: x
  !ERROR: Explicit SAVE of 'y' is redundant due to global SAVE statement
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

subroutine s6
  save x
  save y
  !ERROR: SAVE attribute was already specified on 'y'
  integer, save :: y
  integer, save :: z
  !ERROR: SAVE attribute was already specified on 'x'
  !ERROR: SAVE attribute was already specified on 'z'
  save x,z
end

subroutine s7
  !ERROR: 'x' appears as a COMMON block in a SAVE statement but not in a COMMON statement
  save /x/
end
