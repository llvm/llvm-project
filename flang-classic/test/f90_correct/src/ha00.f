** Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
** See https://llvm.org/LICENSE.txt for license information.
** SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

*   STOP and PAUSE statements.

C    check that STOP and PAUSE statements can be compiled ok, but
C    it's not convenient to check that they actually work ok.

      program p
      data i /7/

      if (i .eq. 0) then
          pause
          PAUSE 0
          pause 00000
          pause 99999
          pause 'Hello'
          stop
      else if (i .lt. 7) then
          stop 1
      elseif  (i .gt. 7) then
          stop 'Fairly long message string'
      else
          call check(77, 77, 1)
      endif

      stop 'this message may or may not be printed'
      end
