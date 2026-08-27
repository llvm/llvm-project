! Test lowering of user-defined recursive function attribute
! RUN: bbc -emit-hlfir -o - %s 2>&1 | FileCheck %s
recursive integer function factorial(n)
    implicit none
    integer, intent(in) :: n
end function factorial

! CHECK: func.func @_QPfactorial{{.*}} attributes {fir.proc_attrs = #fir.proc_attrs<recursive>}
