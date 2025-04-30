** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   VMS IMPLICIT NONE statement.

      program bb10
      implicit none
      integer f
      call check( f(-1), 7, 1)
      end

      function f(a)
      implicitnone
      integer f, a
      f = iabs(-6) + JIABS(a)
      end
