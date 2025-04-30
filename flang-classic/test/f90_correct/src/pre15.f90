!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! This example is slightly modified from the Preprocessor chapter in the
! C99 spec: Example 4 in section 6.10.3.4:
!
! "To illustrate the rules for creating character string literals and
! concatenating tokens..."
!
! Ensure that -Hx,124,0x100000 (Skip Fortran comments) is enabled
!
! Output should be:
! "hello"
! "hello" ", world"
#define glue(a, b)  a ## b
#define xglue(a, b) glue(a, b)
#define HIGHLOW     "hello"
#define LOW         LOW", world"
program p
    logical :: res(1) = .false., expect(1) = .true.
    print *, glue(HIGH, LOW)
    print *, xglue(HIGH, LOW)

    if (glue(HIGH, LOW) == "hello") then
        res(1) = .true.
    endif
    call check(res, expect, 1)
end program
