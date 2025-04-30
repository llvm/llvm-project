! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test TAND intrinsic with quad-precision arguments

program test
  use ieee_arithmetic
  implicit none
  integer, parameter :: n = 16
  integer, parameter :: m = n * 2
  integer, parameter :: k = 16
  real(k), parameter :: pi_q = 3.1415926535897932384626433832795028841_k
  real(k), parameter :: q_tol = 5e-33_k
  real(k), parameter :: degrees(n) = (/ &
     0, & !  0 pi
     30, & ! 1/6 pi
     45, & ! 1/4 pi
     60, & ! 1/3 pi
     90, & ! 1/2 pi
     120, & ! 2/3 pi
     135, & ! 3/4 pi
     150, & ! 5/6 pi
     180, & ! pi
     210, & ! 7/6 pi
     225, & ! 5/4 pi
     240, & ! 4/3 pi
     270, & ! 3/2 pi
     300, & ! 5/3 pi
     315, & ! 7/4 pi
     330  & ! 11/6 pi
  /)
  integer :: i
  real(k), dimension(n) :: arg
  real(k) :: result(m), expect(m)

  expect(1:n) = (/ &
    0.00000000000000000000000000000000000_k, &
   0.577350269189625764509148780501957505_k, &
    1.00000000000000000000000000000000000_k, &
    1.73205080756887729352744634150587251_k, &
   ieee_value(expect(5), ieee_positive_inf), &
   -1.73205080756887729352744634150587251_k, &
   -1.00000000000000000000000000000000000_k, &
  -0.577350269189625764509148780501957505_k, &
    0.00000000000000000000000000000000000_k, &
   0.577350269189625764509148780501957505_k, &
    1.00000000000000000000000000000000000_k, &
    1.73205080756887729352744634150587251_k, &
  ieee_value(expect(13), ieee_negative_inf), &
   -1.73205080756887729352744634150587251_k, &
   -1.00000000000000000000000000000000000_k, &
  -0.577350269189625764509148780501957505_k  &
  /)

  expect(n+1:2*n) = expect(1:n)

  arg = degrees
  result(1:n) = tand(arg)
  result(n+1:2*n) = tand(degrees)

  do i = 1, m
    if (ieee_is_finite(expect(i)) .eqv. .false.) then
      if (ieee_is_finite(result(i)) .eqv. .true.) STOP i
    else if (expect(i) .eq. 0.0_k) then
      if ((result(i) .ne. expect(i)) .and. &
          (abs(result(i)) .gt. q_tol)) STOP i
    else
      if (abs((result(i) - expect(i)) / expect(i)) .gt. q_tol) STOP i
    endif
  enddo

  print *, 'PASS'

end
