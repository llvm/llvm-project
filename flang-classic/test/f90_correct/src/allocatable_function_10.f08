! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Test for allocatable arraies inside functions.

module m
  implicit none

  type t
     integer, allocatable, dimension(:) :: r
  end type t

contains

  function tt(a,b)
    implicit none
    type(t), allocatable, dimension(:) :: tt
    type(t), intent(in), dimension(:) :: a,b
    allocate(tt, source = [a,b])
  end function tt

  function ts(arg)
    implicit none
    type(t), allocatable, dimension(:) :: ts
    integer, intent(in) :: arg(:)
    allocate(ts(1))
    allocate(ts(1)%r, source = arg)
    return
  end function ts

end module m

program test
  use m
  implicit none
  type(t), dimension(2) :: c, d
  c=tt(ts([99,199,1999]),ts([42,142]))
  d=tt(ts([1,2,3]),ts([4,5]))
  call check(c(1)%r, [99,199,1999], 3)
  call check(d(1)%r, [1,2,3], 3)
  call check(c(2)%r, [42,142], 2)
  call check(d(2)%r, [4,5], 2)
end program test
