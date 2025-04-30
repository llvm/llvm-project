!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

! Test fix for kind mismatch when intrinsic len is used in a function call

module str_len_test_mod
 implicit none
 integer :: GLOBAL_SIZE = 100
contains
subroutine caller()
  implicit none
  character(len=GLOBAL_SIZE) :: st
  call called(st, len(st))
end subroutine caller

subroutine called(string, str_len)
  implicit none
  integer, intent(in) :: str_len
  character(len=str_len), intent(inout) :: string
  write(*,*) string, str_len
end subroutine called
end module str_len_test_mod

program test
  use str_len_test_mod
  integer, parameter :: num = 1
  integer rslts(num), expect(num)
  data expect / 1 /

  call caller()

  rslts(1) = 1
  call check(rslts, expect, num)
end program
