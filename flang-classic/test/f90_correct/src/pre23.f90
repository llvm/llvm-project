!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! This tests an empty argument in marcro arguments
!
#define FOO(a, b, c) b, c, 99  /* Ignore first arg             */
#define BAR(a, b, c) a, 98, 99 /* Ignore second and third args */
#define BAZ(a, b, c) a, b, 99  /* Ignore last arg              */
#define QUX(a, b, c) a, b, c   /* Do not ignore any args       */

subroutine dostuff(res, off, x, y, z)
    integer :: res(12)
    integer :: off
    integer :: x, y, z
    print *, x, y, z
    res(off + 0) = x
    res(off + 1) = y
    res(off + 2) = z
end subroutine

program p
    integer :: res(12), expect(12)
    data expect /43, 44, 99, 42, 98, 99, 42, 43, 99, 42, 0, 0/

    call dostuff(res, 1,  FOO(,43,44))     ! Empty first argument
    call dostuff(res, 4,  BAR(42,,))       ! Empty first argument
    call dostuff(res, 7,  BAZ(42,43,))     ! Empty last argument
    call dostuff(res, 10, QUX(42,0,0))     ! All arguments

    call check(res, expect, 12)
end program
