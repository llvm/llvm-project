!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! This tests comments with single or double quotes in the middle, the
! preprocessor should not choke on this ' or " that.
! To enable c99 macro replacement, this should be run with:
!    -Mpreprocess
!    -Hy,124,0x100000 (Do not skip Fortran comments, preprocess them)
!
program p
    logical :: res(1) = .false., expect(1) = .true.
    ! This is a single quote ' that was a single quote, thank you.
    character(32) :: word1 = "Foo" ! This ' is a single quote .

    ! This is a double quote " that was a double quote.
    character(32) :: word2 = "Bar" ! This " is was a double quote.

    if (word1 == 'Foo' .and. word2 == 'Bar') then
        res(1) = .true.
    endif

    call check(res, expect, 1)
end program
