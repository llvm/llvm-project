!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! This tests the warning and line directives.  The warning should be listed as
! occurring on line 42, and then 43, since the #line directive should have
! adjusted the line offset.
!
#line 42
#warning "This is a test (should be line 42...)"
#warning "This is a test (should be line 43...)"
program p
    call check(.true., .true., 1)
end program
