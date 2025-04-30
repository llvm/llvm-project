! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test MINEXPONENT intrinsic with boundary values

program exponentmin
    implicit none
    integer :: n = -16381
    integer, parameter :: m = 7
    real(kind=16) :: gig = 1._16/0._16
    real(kind=16) :: nan = 0._16/0._16
    real(kind=16) :: zero = +0._16
    real(kind=16) :: minus_gig = -1._16/0._16
    real(kind=16) :: nonstdmin = 6.5e-4966_16
    real(kind=16) :: nonstdmax = 1.7e-4932_16
    real(kind=16) :: minus_zero = -0._16
    logical, dimension(m) :: result, expect

    expect = .true.
    result = .false.

    result(1) = minexponent (gig) .eq. n
    result(2) = minexponent (nan) .eq. n
    result(3) = minexponent (zero) .eq. n
    result(4) = minexponent (minus_gig) .eq. n
    result(5) = minexponent (nonstdmin) .eq. n
    result(6) = minexponent (nonstdmax) .eq. n
    result(7) = minexponent (minus_zero) .eq. n

    call check(result, expect, m)

end program exponentmin
