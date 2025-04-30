!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! This tests use of a 'c' as a macro parameter.  'c' should not be detected as
! a comment.
!
#define FOO(c) c + 2
program p
    integer :: expected(1) = 42, res(1) = FOO(40)
    call check(res, expected, 1)
end program
