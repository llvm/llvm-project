! RUN: %python %S/../test_errors.py %s %flang -fopenacc
integer function square(x)
    implicit none
    integer, intent(in) :: x
!ERROR: logical expression or object list expected
    !$acc parallel self(,)
    !$acc end parallel
!ERROR: logical expression or object list expected
    !$acc parallel self(.true., )
    !$acc end parallel
    square = x * x
end function square
