! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test MAXEXPONENT intrinsic with quad-precision arguments

program exponentmax
    implicit none
    real(kind=16) :: args1 = 123.12_16
    integer :: args2 = maxexponent(args1)
    integer, parameter :: args3 = maxexponent(args1)

    call test (args2, args3, maxexponent(args1), maxexponent(123.12_16))

end program exponentmax

!Check result
subroutine test (arg1, arg2, arg3, arg4)
    implicit none
    integer :: n = 16384
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
