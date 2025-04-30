!
** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   VMS end-of-line comments and debug statements.

      program ! continue . . .
c . . .
     +        ab10   ! end of program statement

      integer rslts(3), expect(3)!
	! (tab)
      data expect / 1, 2, 3 /!/('
 !      x
  !     x
   !    x
    !   x
      ! x
      rslts(1) = 1	!" 
D     rslts(1) = 10
D    !
      rslts(2) = 2
D9999 rslts(2) =
D    + 22 ! x
      
      rslts(3) = 3
      call check(rslts, expect, 3)
      !END
      end!
!
D
D     subroutine ss
D     end
D
      subroutine ss
      character c
      c = '!'
      i = 4h!!!!
      end
!  last line of file is a comment ...
