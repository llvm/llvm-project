! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp

module new_operator
    implicit none

    interface operator(.MYOPERATOR.)
       module procedure myprocedure
    end interface
contains
    pure integer function myprocedure(param1, param2)
        integer, intent(in) :: param1, param2
        myprocedure = param1 + param2
    end function
end module

program sample
    use new_operator
    implicit none
    integer :: x, y 

    !$omp atomic update
        x = x / y
     
    !$omp atomic update
    !ERROR: A call to this function is not a valid ATOMIC UPDATE operation
        x = x .MYOPERATOR. y

    !$omp atomic
    !ERROR: A call to this function is not a valid ATOMIC UPDATE operation
        x = x .MYOPERATOR. y
end program
