!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
      module trailz_elemental
         implicit none
         contains
            elemental integer  function do_trailz(a) result(b)
               integer, intent (in) ::  a
               b=trailz(a)
            end function
      end module


      program test
      use trailz_elemental
       implicit none
       integer, parameter :: num = 5
       integer results(num)
       integer , parameter :: expect(num) =(/2,6,6,0,0/)
       integer , parameter :: arr(num)=(/-108,-64,64,-1,1/)

       results=do_trailz (arr)

       call check(results, expect, num)
      end program
