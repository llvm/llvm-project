!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
module mod_submod32
integer a, b, m
interface
    module subroutine check_arr(arr)
        integer, intent(in) :: arr(:)
    end subroutine
end interface
end module mod_submod32


submodule (mod_submod32) submod_submod32
contains
    module procedure check_arr
        a = lbound(arr, DIM=1)
        b = size(arr)
        m = maxval(arr)
    end procedure
end submodule submod_submod32

program prog
    use mod_submod32
    integer x(7:15)
    x(7:15) = 0
    x(9:10) = 1
    x(15) = 10
    print *, "lbound: ", lbound(x)
    print *, "kind: ", kind(x)
    print *, "maxval: ", maxval(x)
    call check_arr(x)
    if ( a .EQ. 1 .AND. lbound(x, DIM=1) .EQ. 7 .AND. b .EQ. size(x) .AND. maxval(x) .EQ. m) then
        print *, " PASS "
    else 
        print *, "FAILED: lbound of arr in submod is ", a, " and the expection is 1"
        print *, "FAILED: lbound of x is is ", lbound(x, DIM=1), "and the expection is 7"
        print *, "FAILED: size of arr in submod is ", b, " and size of x is ", size(x)
        print *, "FAILED: maxval of arr in submod is", m, " and maxval of x is", maxval(x)
    end if
end program prog
