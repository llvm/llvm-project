!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module t3649
    type tb_config
        double precision, pointer :: pos(:,:)
    end type tb_config
contains
    integer function allo_wrap_f(a, n, m)
    implicit none
        double precision, pointer :: a(:,:)
        integer n,m
        a(1:n,1:m) = 16
        allo_wrap_f = -1
    end function
    subroutine allo_wrap_s(a, n, m)
    implicit none
        double precision, pointer :: a(:,:)
        integer n,m
        a(1:n,1:m) = 32
    end subroutine
end module t3649

program test
use t3649
implicit none
    type(tb_config) a
    double precision, pointer :: a_pos(:,:)
    integer st
    integer, parameter :: N = 3
    integer :: result(N)
    integer, save :: expect(N) = (/64,32,32/)

    allocate(a%pos(2744,3))
!    print *, "call allo_wrap_s on a%pos"
    call allo_wrap_s(a%pos,2744,3)
    result(1) = a%pos(1,1) + a%pos(2744,3)
!    print *, "done"
    deallocate(a%pos)

    allocate(a_pos(2744,3))
!    print *, "call allo_wrap_f on a_pos"
    st = allo_wrap_f(a_pos,2744,3)
    result(2) = a_pos(1,1) + a_pos(2744,3)
!    print *, "done"

    allocate(a%pos(2744,3))
!    print *, "call allo_wrap_f on a%pos"
    st = allo_wrap_f(a%pos,2744,3)
    result(3) = a%pos(1,1) + a%pos(2744,3)
!    print *, "done"
    call check(result, expect, N)

end program
