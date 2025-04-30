! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test I/O of quad-precision values

program test
  integer, parameter :: qp = 16, n = 5
  complex(kind = qp) :: c(n)
  integer :: result(n), expect(n), i
  expect = 1
  result = 0
  c(1) = (1.23456789123456789123456789123456789e+9_qp,&
          -1.23456789123456789123456789123456789e+99_qp)
  c(2) = (1.23456789123456789123456789123456789e-33_qp,&
          -1.23456789123456789123456789123456789e-333_qp)
  c(3) = cmplx(huge(0.0_qp), -huge(0.0_qp), kind = qp)
  c(4) = cmplx(tiny(0.0_qp), -tiny(0.0_qp), kind = qp)
  c(5) = cmplx(epsilon(0.0_qp), -epsilon(0.0_qp), kind = qp)

  do i = 1, n
    write(*, 100) c(i)
  enddo

  100 FORMAT('', EN46.28E4, EN46.28E4)
  print *, 'PASS'
end program
