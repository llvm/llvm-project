!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test concatenation and to ensure the preprocessor does not strip the
! concatenation operator, but also expands the items following '//'
!
#define STR  "bar"
#define STR2 "baz"
program p
    logical :: res(1) = .false., expect(1) = .true.
    print *, "foo"//STR//STR2
    if ("foo"//STR//STR2 == "foobarbaz") then
        res(1) = .true.
    endif

    call check(res, expect, 1)
end program
