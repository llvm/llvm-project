** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   POINTER statements - adjustable arrays, loc intrinsic

      parameter(N = 13)
      integer result(N), expect(N)
      pointer (ptr, idum)

      data expect /
     +  2, 3, 3, 4, 			! tests 1-4, ment
     + 2, 3, 4, 3, 4, 5, 4, 5, 6	! tests 5-13, ent
     + /

      ptr = loc(result(1))
      call ment(ptr, 2)

      ptr = loc(result(5))
      call ent(ptr, 3)

      call check(result, expect, N)
      end

      subroutine ment(p, n)
      entry foo(p, n)
      dimension iarr(n, n)
      pointer (p, iarr)	! object refd by common arg declared after entry
      do i = 1, n
	  do j = 1, n
	      iarr(i, j) = i + j
	  enddo
      enddo
      return
      end
      subroutine bar(p, n)
      pointer (p, iarr)	! object refd by common arg declared before entry
      integer iarr(n, n)
      entry ent(p, n)
      do i = 1, n
	  do j = 1, n
	      iarr(i, j) = i + j
	  enddo
      enddo
      return
      end
