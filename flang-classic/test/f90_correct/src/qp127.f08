! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test GAMMA and LOG_GAMMA intrinsics with quad-precision arguments

program gamma_test
  implicit none
  intrinsic :: gamma, log_gamma
  integer, parameter :: qp = selected_real_kind(precision (0.0_8) + 1)

  real(qp) :: rqp

  if (abs(gamma(1.0_qp)  - 1.0_qp) > tiny(1.0_qp)) STOP 1
  if (abs(log_gamma(1.0_qp)) > tiny(1.0_qp)) STOP 2
  write(*,*) 'PASS'
end program gamma_test

