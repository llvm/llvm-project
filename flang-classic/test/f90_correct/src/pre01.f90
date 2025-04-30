!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! This tests macro expansion with single and double quoted characters
!
#define STR1 "Don't"
#define STR2 'Don''t'
#define STR3 "Don""t"
#define STR4 'Don"t'
#define FOO(_str) _str

program p
    logical :: res(3) = (/.false., .false., .false./)
    logical :: expect(3) = (/.true., .true., .true./)
    character(32) :: word1 = STR1
    character(32) :: word2 = STR2
    character(32) :: word3 = STR3
    character(32) :: word4 = STR4
    character(32) :: word5 = FOO('foo "" bar')
    character(32) :: word6 = FOO("foo '' bar")
    print *, word1
    print *, word2
    print *, word3
    print *, word4
    print *, word5
    print *, word6

    if (word1 == "Don't" .and. word2 == "Don't") then
        res(1) = .true.
    endif
    if (word3 == 'Don"t' .and. word4 == 'Don"t') then
        res(2) = .true.
    endif

    if (word5 == 'foo "" bar' .and. word6 == "foo '' bar") then
        res(3) = .true.
    endif
    
    call check(res, expect, 3)
end program
