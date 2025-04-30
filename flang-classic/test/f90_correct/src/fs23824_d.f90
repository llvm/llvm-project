! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

module test_module
implicit none
private :: c_loc
contains
  integer function c_loc(value) result(res)
    integer :: value
    res = value
  end function
end module

program test_c_loc
  use test_module
  integer :: value = 2
  integer :: cptr = 0
  cptr = c_loc(value)
  write(*,*) "cptr = ", cptr
end program
