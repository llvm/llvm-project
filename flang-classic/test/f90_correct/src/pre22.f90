!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! This tests commas in replacement lists
!
#define FOO(a, b, c) a, b, c

subroutine dostuff(x, y, z)
    integer :: x, y, z
    if (x .eq. 42 .and. y .eq. 43 .and. z .eq. 44) then
        call check(.true., .true., 1)
    else
        call check(.false., .true., 1)
    endif
    print *, x, y, z
end subroutine

program p
    call dostuff(FOO(42, 43, 44))
end program
