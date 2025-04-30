! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!  need to allocate temp of dynamic size for the call when
!  concatenating two assumed-length character dummy variables
!
      subroutine something(a)
      character*(*) a
      integer result(0:10)
      integer expect(0:10)
      data expect/10,116,104,105,115,32,116,104,97,116,32/

      result(0) = len(a)
      do i = 1,min(10,len(a))
       result(i) = ichar(a(i:i))
      enddo
      !print *,result
      call check(result,expect,11)
      end
      subroutine concat(c1,c2)
      character(*) :: c1, c2
      call something(c1//c2)
      end

      call concat( 'this',' that ')
      end
