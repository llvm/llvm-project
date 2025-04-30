!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

program gppkernel

!      implicit none
      integer :: ngpown
!      integer :: my_igp


!$OMP PARALLEL  
       ngpown = 2
!$OMP END PARALLEL

!$OMP PARALLEL  default(firstprivate)
      ngpown = 1
!$OMP END PARALLEL

    print *, "PASS"

end program
