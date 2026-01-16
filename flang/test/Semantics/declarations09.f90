! RUN: %python %S/test_errors.py %s %flang_fc1
program test
        type :: easy
        integer :: i
        end type
        type :: check
        procedure(class(easy)),pointer, nopass :: X10      !ok
        end type
end program 