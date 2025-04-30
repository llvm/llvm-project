
** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

**--- INQUIRE of preconnected units

      parameter (N=2)
      character*10 buf
      integer result(N), expect(N)
      logical is_open

      INQUIRE(unit=5, opened=is_open)
      result(1) = and(is_open, 1)
      INQUIRE(unit=6, opened=is_open)
      result(2) = and(is_open, 1)

      data expect/1, 1/
      call check(result, expect, N)
      end
