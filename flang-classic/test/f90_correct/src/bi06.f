!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! bind(c) complex function -- -O delete_stores() failure; derived from the
!    entry test, ie10.f
       complex function cp1(i, j) bind(c)
       cp1 = cmplx(i, j)
       cp1 = cp1 + (1, 1)
       end
       interface 
         complex function cp1(i, j) bind(c)
         endfunction
       endinterface
       integer(4) expect(2)
       integer(4) res(2)
       data expect/2,4/
       complex z
       z = cp1(1,3)
       res(1) = real(z)
       res(2) = aimag(z)
!       print *, z
       call check(res, expect, 2)
       end
