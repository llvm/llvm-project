
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test that output of slice works o.k.
        PROGRAM test

           IMPLICIT NONE

        integer result(3),expect(3)
        data expect/2,4,6/


        integer a(2,3)
        integer res(3)

        
        a(1,1) = 1
        a(2,1) = 2
        a(1,2) = 3
        a(2,2) = 4
        a(1,3) = 5
        a(2,3) = 6

        OPEN (UNIT=10, FILE = "io02.txt", STATUS = "REPLACE")
        write(UNIT=10,*)  a(2:2,:)
        CLOSE (10)
        OPEN (UNIT=10, FILE = "io02.txt", STATUS = "OLD")
        read (10,*) result(:)
        CLOSE (10)
        call check(result,expect,3)
        end program

!           2            4            6
