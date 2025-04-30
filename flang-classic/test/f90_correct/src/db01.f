** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   IACHAR intrinsics.
      module db01
      integer(4), parameter :: N = 10
      integer(4) ix
      integer(4) result(N)
      integer(4) expect(N)
      data expect/97,122,98,121,99,120,100,119,101,118/
      contains
      subroutine dummy(i)
        integer(8)  i
        ix = ix + 1
        result(ix) = i
      endsubroutine
      subroutine test1
        implicit none
        integer(8), parameter :: icode1 = iachar("a")
        integer(8), parameter :: icode2 = iachar("z")
        call dummy(icode1)
        call dummy(icode2)
      endsubroutine
      subroutine test2
        implicit none
        integer(8), parameter :: icode1 = iachar('b', 8)
        integer(8), parameter :: icode2 = iachar('y', 8)
        call dummy(icode1)
        call dummy(icode2)
      endsubroutine
      subroutine test3(c1,c2)
        implicit none
        character*1 c1, c2
        integer(8) icode1
        integer(8) icode2
        icode1 = iachar(c1,8)
        icode2 = iachar(c2,4)
        call dummy(icode1)
        call dummy(icode2)
      endsubroutine
      subroutine test4(c1,c2)
        implicit none
        character*1 c1, c2
        integer(8) icode1
        integer(8) icode2
        icode1 = iachar(c1,8)
        icode2 = iachar(c2,4)  !! yes, I chose 4
        call dummy(icode1)
        call dummy(icode2)
      endsubroutine
      subroutine test5()
        implicit none
        integer(8) :: icode1 = iachar('e',8)
        integer(8) :: icode2 = iachar('v')
        call dummy(icode1)
        call dummy(icode2)
      endsubroutine
      end 
      use db01
      call test1
      call test2
      call test3('c','x')
      call test4('d','w')
      call test5
      call check(result, expect, N)
      end
