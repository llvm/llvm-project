!* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
!* See https://llvm.org/LICENSE.txt for license information.
!* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!       simple parallel do, negative stride

program main
  call mmm(10, 1, -2)
contains

subroutine mmm (m1, m2, m3)
  integer :: m, m1, m2, m3
!$OMP PARALLEL DO
  do m = m1, m2, m3
    print *, "PASS"
  enddo
!$OMP END PARALLEL DO
end subroutine

end program

