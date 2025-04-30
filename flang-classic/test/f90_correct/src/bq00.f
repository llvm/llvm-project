** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   POINTER statements

      parameter(N = 5)
      integer result(N), expect(N)
      double precision dum(2)
      double precision d
      pointer(ptr, d)
      double complex cd
      pointer(ptr, cd)
      pointer (p1, ib)
      pointer (p2, p1)

      data expect /
     +  2,
     +  4,
     +	2, 4,
     +  10
     + /

      p2 = loc(result(5))
      ptr = loc(dum)
      dum(1) = 2.0
      dum(2) = 4.0

      result(1) = d

      ptr = ptr + 8
      result(2) = d

      ptr = ptr - 8
      result(3) = real(cd)

      result(4) = dimag(cd)

      ib = 10		! two levels of pointer. should be result(5)

      call check(result, expect, N)
      end
