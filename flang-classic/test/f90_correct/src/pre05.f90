!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! This tests token pasteing (aka concatenation)
!
#define CMD(_prefix, _fname, _msg, _res) _prefix ##_fname(_msg, _res)

subroutine foobar(words, res)
    logical :: res(1)
    character(12) :: words
    print *, words
    if (words == "G'day! Mate!") then
        res(1) = .true.
    endif
end subroutine

program p
    logical :: res(1) = .false., expect(1) = .true.
    call CMD(foo, bar, "G'day! Mate!", res)
    call check(res, expect, 1)
end program
