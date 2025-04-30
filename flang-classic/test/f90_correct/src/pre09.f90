!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! This tests stringization
!
#define STR(_s) #_s

program p
    logical :: res(1) = .false., expect(1) = .true.
    print *, STR(A quote "quote" and 'another' in the middle! ... Of a string!)
!    print *, STR(This is an awesome example of string replacement)
    if (STR(foobarbaz) == 'foobarbaz') then
        res(1) = .true.
    endif

    call check(res, expect, 1)
end program
