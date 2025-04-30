! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! test for removing deep dealloction of temp variables.
program test
  implicit none

  type t
      integer, allocatable, dimension(:) :: r
  end type t

  type(t), allocatable, dimension(:) :: arg1, arg2
  type(t), dimension(2) :: d

  allocate(arg1(1))
  allocate(arg1(1)%r, source=[99,199,1999])
  allocate(arg2(1))
  allocate(arg2(1)%r, source=[42,142])
  d = [arg1, arg2]
  deallocate(arg1)
  deallocate(arg2)

  call check(d(1)%r, [99,199,1999], 3)
  call check(d(2)%r, [42,142], 2)
end program test
