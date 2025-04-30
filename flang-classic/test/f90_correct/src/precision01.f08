! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test PRECISION intrinsic with quad arguments

program precisiontest     
    implicit none
    real(kind=16) :: a = 1.2e+33_16
    integer :: b = precision(a)
    integer, parameter :: c = precision(2.22e+36_16)

    call test (b, c, precision(a), precision(2.5e+35_16))
end program precisiontest


subroutine test (arg1, arg2, arg3, arg4)
    implicit none
    integer :: n = 33
    integer, parameter :: m = 4
    integer, intent(in) :: arg1, arg2, arg3, arg4
    logical, dimension(m) :: result, expect

    expect = .true.
    result = .false.

    result(1) = arg1 .eq. n
    result(2) = arg2 .eq. n
    result(3) = arg3 .eq. n
    result(4) = arg4 .eq. n
    
    call check(result, expect, m)
end subroutine test
