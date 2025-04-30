!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! This tests stringization
!
#define _TURTLES2(_msg) _msg##"And down again..."
#define _TURTLES1(_msg) _TURTLES2(_msg ## "And down...")
#define TURTLES(_msg) _TURTLES1(_msg)

program p
    logical :: res(1) = .false., expect(1) = .true.
    print *, TURTLES("All the way down!")
    if (TURTLES("All the way down!") == 'All the way down!"And down..."And down again...') then
        res(1) = .true.
    endif

    call check(res, expect, 1)
end program
