** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   END DO statement (VMS).

      program hb30
      parameter (N = 7)
      integer*2 i
      integer j
      integer rslts(N), expect(N)

      do 10 i = 1, 10, 100
          rslts(1) = i
          data expect(1) /1/
10    enddo

      rslts(2) = 0
      do i = -10, -11, -1
          rslts(2) = rslts(2) + i
      END DO
      data expect(2) / -21 /

      do while (0 .gt. 1)
      enddo

      rslts(3) = 3
      do while ( rslts(3) .lt. 10 )
          if (rslts(3) .eq. -1)  goto 30
              rslts(3) = rslts(3) * rslts(3)
30    end do
      data expect(3) / 81 /

      j = 1
      do 40 while (j .ne. 3)
          j = j + 1
          do i = j, j+1
              rslts(i+j) = i + j
 40   END
     +DO
      data (expect(j), j = 4, 7) / 4, 5, 6, 7 /

      call check(rslts, expect, N)
      end
