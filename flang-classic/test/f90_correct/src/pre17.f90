!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test multi-line and stringification and concatenation directives
!
#define STR(_s) # _s
#define DEF character(32) ::
#define DEFSTR(_s) DEF _s \
    = \
    #_s

#define MKSTR(_a, _b) DEF _a ## _b \
    = \
    STR(_a ## _b)

program p
    logical :: res(1) = .false., expect(1) = .true.
    DEFSTR(foo)
    MKSTR(foo, bar)
    print *, foo
    print *, foobar

    if (foo == 'foo' .and. foobar == 'foobar') then
        res(1) = .true.
    endif

    call check(res, expect, 1)
end program
