!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for array constructor whose element is array of dynamic char.

program m_test_adjust_len
  call test_adjust_len(4, 4, 6)
  print *, "PASS"

contains
  subroutine test_adjust_len(a, b, c)
    integer :: a, b, c
    character(len=a) :: s1(2)
    character(len=b) :: s2
    character(len=c) :: s3(1)

    s2 = "aaaa"
    s3 = "bbbbbb"

    s1 = [s2, s3]

    if (s1(1) /= "aaaa") stop 1
    if (s1(2) /= "bbbb") stop 2
  end
end
