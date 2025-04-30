!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

! RUN: %flang %s
!! This test is a reproducer of a crash bug in the dinit_data function in
!! flang2/flang2exe/dinit.cpp. The crash is caused by ivl being NULL.
program dinit_crash
    implicit none
    type struct1
    end type struct1
    type struct2
        integer :: i
    end type struct2
    type struct3
        type(struct2) :: f1
        type(struct1) :: f2
    end type struct3
    type(struct3) :: v = struct3(struct2((-2)), struct1())
    print *, "OK"
end program dinit_crash
