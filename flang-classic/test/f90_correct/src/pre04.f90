!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! This tests stringization
! The idea of this came from the C99 spec
!
#define hash_hash # ## #
#define mkstr(_str) #_str
#define in_between(_z) mkstr(_z)
#define join(_x, _y) in_between(_x hash_hash _y)

program p
    logical :: res(1) = .false., expect(1) = .true.
    print *, join(foo, bar)
    if (join(foo, bar) == 'foo ## bar') then
        res(1) = .true.
    endif

    call check(res, expect, 1)
end program
