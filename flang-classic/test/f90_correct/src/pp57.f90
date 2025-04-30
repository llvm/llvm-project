! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! flang#196 ICE: mk_mem_ptr_shape: lwb not sdsc
module t4
  implicit none

  ! error happened because both type have member named 'm'
  type T1
    integer, dimension(:,:), allocatable :: m
    integer, dimension(:,:), allocatable :: m1
  end type
  type T2
    type(T1), dimension(:,:), allocatable :: m
    type(T1), dimension(:,:), allocatable :: m2
  end type

contains

  subroutine sub(x)
    type(T2) :: x
    type(T1), dimension(:,:), allocatable :: tmp
    integer :: i = 1
    allocate(tmp(i,i))
    call move_alloc(tmp(i,i)%m1, x%m2(i,i)%m)
  end subroutine

end module
