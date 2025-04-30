!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
module foo_submod29
integer, allocatable :: arr
interface
    module subroutine check_alloc
    end subroutine
    module subroutine check_not_alloc
    end subroutine
end interface
end module

submodule (foo_submod29) bar
contains
    module procedure check_alloc
        if ( allocated(arr) ) then
            print *, " PASS "
        else
          print *, "FAIL"
        endif
    end procedure
    module procedure check_not_alloc
        if ( .not. allocated(arr) ) then
            print *, "PASS"
        else
          print *, "FAIL"
        endif
    end procedure
end submodule

program foobar
use foo_submod29
implicit none
call check_not_alloc
allocate (arr)
call check_alloc
end program foobar
