! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test HYPOT intrinsic with quad-precision arguments

program p
  use ISO_C_BINDING
  use check_mod

  interface
    subroutine get_expected_q( src1, src2, expct, n ) bind(C)
      use ISO_C_BINDING
      type(C_PTR), value :: src1
      type(C_PTR), value :: src2
      type(C_PTR), value :: expct
      integer(C_INT), value :: n
    end subroutine
  end interface

  integer, parameter :: N=20
  real*16, target,  dimension(N) :: q_src1
  real*16, target,  dimension(N) :: q_src2
  real*16, target,  dimension(N) :: q_rslt
  real*16, target,  dimension(N) :: q_expct

  real*16 :: q_eps = 1.0e-33_16

  do i =  0, N-1
    q_src1(i+1) = .14*i +  i+1
    q_src2(i+1) = .14*i - i+N
  end do

  q_rslt = hypot(q_src1, q_src2)

  call get_expected_q(C_LOC(q_src1), C_LOC(q_src2), C_LOC(q_expct), N)

  call checkr16( q_rslt, q_expct, N, rtoler=q_eps)
end program
