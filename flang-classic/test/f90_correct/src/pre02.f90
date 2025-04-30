!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! This tests macro replacement
!
#define PR(_expr, _msg) if (_expr) then; print *, _msg; res(1)=42; end if

program p
    integer :: x
    integer :: res(1) = 0, expect(1) = 42
    PR(100 .lt. 200, "Yes!")
    call check(res, expect, 1)
end program
