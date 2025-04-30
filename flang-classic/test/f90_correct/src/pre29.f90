
!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! This tests pointer '=>' in macro bodies.
!
#define SET(p, t) p => t
program p
    integer, target ::  a = 10
    integer, pointer :: pa
    integer :: res(1) = 0, expect(1) = 10
    SET(pa, a)
    res(1) = pa
    call check(res, expect, 1)
end program
