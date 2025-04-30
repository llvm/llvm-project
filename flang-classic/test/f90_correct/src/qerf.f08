! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test error function intrinsics (ERF/ERFC) with quad-precision arguments

program p
  integer, parameter :: n = 4
  integer, parameter :: k = 16
  real(16) :: rslts(n), expect(n)
  real(kind = k) :: t1
  real(kind = k) :: eps_q = 1.e-33_16
  integer :: i
  t1 = 1.0_16 
  expect(1) = 0.842700792949714869341220635082609264_16 
  expect(2) = 0.842700792949714869341220635082609264_16
  expect(3) = 0.157299207050285130658779364917390736_16
  expect(4) = 0.157299207050285130658779364917390736_16
  
  rslts(1) = erf(t1)
  rslts(2) = erf(1.0_16)
  rslts(3) = erfc(t1)
  rslts(4) = erfc(1.0_16)

  do i = 1, n
    if (abs((rslts(i) - expect(i)) / expect(i)) > eps_q) STOP i
  enddo
  
  print *, 'PASS' 
end program p
