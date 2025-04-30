! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test inverse hyperbolic intrinsics (ASINH/ACOSH/ATANH) with quad-precision arguments

program p
  integer, parameter :: n = 6
  integer, parameter :: k = 16
  real(kind = k) ::  expect(n)
  real(kind = k) :: t1
  real(16) :: eps_q = 1.e-33_16
  integer(4) :: flag(n), expf(n)

  flag = 0
  expf = 1
  expect(1) = 1.01053731674044269796875066695517919_16
  expect(2) = 0.626373448569271674938017357627804168_16
  expect(3) = 0.881762442279148179873159768470534111_16
  expect(4) = 1.31695789692481093512235445103107846_16
  expect(5) = 15.3133766947412446008256083103178189_16
  expect(6) = 0.110883748300181152379133826649413080_16

    t1 = 1.55555_16
    if (abs((acosh(t1) - expect(1)) / acosh(t1)) <= eps_q) flag(1) =1
    t1 = 0.55555_16
    if (abs((atanh(t1) - expect(2)) / atanh(t1)) <= eps_q) flag(2) =1
    t1 = 1.00055_16
    if (abs((asinh(t1) - expect(3)) / asinh(t1)) <= eps_q) flag(3) =1
    if (abs((acosh(1.99999999999999_16) - expect(4)) / acosh(1.99999999999999_16)) <= eps_q) flag(4) =1
    if (abs((atanh(0.9999999999999_16) - expect(5)) / atanh(0.9999999999999_16)) <= eps_q) flag(5) =1
    if (abs((asinh(0.11111111111_16) - expect(6)) / asinh(0.11111111111_16)) <= eps_q) flag(6) =1

  call check(flag, expf, n)

end program p
