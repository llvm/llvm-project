!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! This tests stringization
! Turn 'a' into a string
!
#define FOO(a, b) #a
program p
    logical :: res(1) = .false., expect(1) = .true.
    print *, FOO(Hello, Ignore)

    if (FOO(Hello, Ignore) == 'Hello') then
        res(1) = .true.
    endif

    call check(res, expect, 1)
end program
