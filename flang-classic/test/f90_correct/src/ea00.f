** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   Test data initializations of scalar variables of type
*   integer, logical, integer*2, and logical*1.

      program xx
      
      common /i/ i, j, si, k, sj, sk
      integer i, j, k
      integer *2  si, sj, sk

      common /logical/ l, sl, m, sm
      common /logical/ sn, n
      logical l, m, n
      logical*1 sl, sn, sm

      data i /2147483647/, j /-2147483647/ si/ +3/
      data  sj, sk, k /  32767, -32768, -00/

      data l, sl, m, n / .TRUE., .true., .FALSE., .TRUE. /
      data  sm, sn / .false., .true.  /

      call mycheck()
      end

      subroutine mycheck

      parameter (NN = 12)
      integer rslts(NN), expect(NN)
      logical lrslts(NN)
      equivalence (rslts(1), lrslts(1))
      
      common /i/ i, j, si, k, sj, sk
      integer i, j, k
      integer *2  si, sj, sk

      common /logical/ l, sl, m, sm
      common /logical/ sn, n
      logical l, m, n
      logical*1 sl, sn, sm

      rslts(1) = i
      rslts(2) = j
      rslts(3) = k
      rslts(4) = si
      rslts(5) = sj
      rslts(6) = sk
      lrslts(7) = l
      lrslts(8) = m
      lrslts(9) = n
      lrslts(10) = sl
      lrslts(11) = sm
      lrslts(12) = sn

      data expect / 2147483647, -2147483647, 0, 3, 32767, -32768,
     +             -1, 0, -1, -1, 0, -1       /

      call check(rslts, expect, NN)
      end
