! RUN: %python %S/../test_errors.py %s %flang -fopenacc -pedantic

module mm_acc_rout_function
contains
    integer function dosomething(res)
        !$acc routine seq
        integer :: res
        dosomething = res + 1
    end function
end module
    
program main
    use mm_acc_rout_function
    implicit none
    integer :: res = 1
    !$acc serial default(none) copy(res)
    res = dosomething(res)
    !$acc end serial
end program

