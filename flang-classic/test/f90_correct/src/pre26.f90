!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! This tests __VA_ARGS__ replacement in macros
!
#define PR(...) print *, "Test:", __VA_ARGS__
#define CPY(...) __VA_ARGS__
program p
    integer res(4), expect(4);
    data res / CPY(1,2,3),4 /
    data expect / 1,2,3,4 /
    PR("This is a string with values", 1,2,3)
    call check(res, expect, 4)
end program
