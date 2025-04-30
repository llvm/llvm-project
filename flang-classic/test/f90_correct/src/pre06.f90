!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! This tests replacement inside directives
!
#define FillInDirective FLUSH

program p
    logical :: res(1) = .false., expect(1) = .true.
    !$omp FillInDirective
    print *, "Testing a single quote ' yep! That's a quote alright"
    res(1) = .true.
    call check(res, expect, 1)
end program
