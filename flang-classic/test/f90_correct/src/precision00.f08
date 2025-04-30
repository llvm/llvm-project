! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test PRECISION intrinsic with quad and qcmplx arguments

program precisiontest
    implicit none
    integer :: n = 33
    integer, parameter :: m = 9
    real(kind=16) :: a = 12.3_16
    real(kind=16) :: b = +0._16       ! +0
    real(kind=16) :: c = -0._16       ! -0
    real(kind=16) :: d = 1._16/0._16  ! +inf
    real(kind=16) :: e = 0._16/0._16
    real(kind=16) :: f = -1._16/0._16 ! -inf
    real(kind=16) :: g = 6.5e-4966_16 ! 0x0000000000000001, approximately
    complex(kind=16) :: y, z
    logical, dimension(m) :: result, expect

    y = cmplx(1._16, 0._16)
    z = cmplx(e, d)

    expect = .true.
    result = .false.

    result(1) = precision(a) .eq. n
    result(2) = precision(b) .eq. n
    result(3) = precision(c) .eq. n
    result(4) = precision(d) .eq. n
    result(5) = precision(e) .eq. n
    result(6) = precision(f) .eq. n
    result(7) = precision(g) .eq. n
    result(8) = precision(y) .eq. n
    result(9) = precision(z) .eq. n

    call check(result, expect, m)
end program precisiontest
