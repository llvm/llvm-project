! RUN: %python %S/../test_errors.py %s %flang -fopenacc
integer function square(x, y)
    implicit none
    integer, intent(in) :: x, y
    !$acc parallel self(x * 2 > x) ! ok
    !$acc end parallel
!ERROR: Must have LOGICAL type, but is INTEGER(4)
    !$acc parallel self(x * 2)
    !$acc end parallel
!ERROR: SELF clause on the PARALLEL directive only accepts optional scalar logical expression
    !$acc parallel self(x, y)
    !$acc end parallel
    square = x * x
end function square
