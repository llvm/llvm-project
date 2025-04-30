!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!   passing nocontiguous section to subroutine

      program main

      implicit none

      integer xyz(2,4)
      integer result(4),expect(4)
      data expect/ 1,2,3,4/

      xyz(1,1) = 1
      xyz(1,2) = 2
      xyz(1,3) = 3
      xyz(1,4) = 4
      xyz(2,1) = 5
      xyz(2,2) = 6
      xyz(2,3) = 7
      xyz(2,4) = 8

      call junk(xyz(1,:), 4, result)

      call check(result,expect,4)

      end


      subroutine junk(abc, num, res)
      integer num,abc(num),res(4)
      res(:) = abc(1:4)
      end

