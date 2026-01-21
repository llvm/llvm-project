! RUN: %flang_fc1 %s 2>&1 | FileCheck %s

program test_exit
    implicit none
    integer(8) :: status
    integer :: status_default
    
    status = 0
    call exit(status)
    ! CHECK: portability: 'exit' intrinsic converts INTEGER(kind=8)
    
    status_default = 0
    call exit(status_default)
    
end program test_exit
