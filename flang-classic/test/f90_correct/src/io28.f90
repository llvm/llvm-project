! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for I/O take derived type array reference.
! The function should be called only one time if a function
! reference appears in the derived type array reference.

program test
  type my_type
     integer :: i
     integer :: j
  end type my_type
  character(80) :: str1(2), str2(2)
  type(my_type) :: t(2)

  t(1) = my_type(11, 12)
  t(2) = my_type(21, 22)

  write(str1(1), *), t(1)
  write(str1(2), *), t(2)
  write(str2(1), *), t(func())
  write(str2(2), *), t(func())
  if (any(str1 .ne. str2)) STOP 1
  write(*, *) 'PASS'

contains
  integer function func()
    integer, save :: i = 1
    func = i
    i = i + 1
  end function
end
