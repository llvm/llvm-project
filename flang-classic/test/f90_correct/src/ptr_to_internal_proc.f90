!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module m
  implicit none

  interface
    subroutine int_printer(i)
      integer, dimension(:), intent(in) :: i
    end subroutine
  end interface
  procedure(int_printer), pointer :: iprinter

contains
  subroutine work
    implicit none
    integer, dimension(:), allocatable :: i1

    allocate(i1(2))
    call iprinter(i1)
    deallocate(i1)
  end subroutine
end module

subroutine sub
  use m
  implicit none
  integer :: num_prints
  integer, parameter :: expect = 1

  num_prints = 0
  iprinter => myprint
  call work
  call check(num_prints, expect, 1)
contains
  subroutine myprint(i)
    integer, dimension(:), intent(in) :: i

    print *, i
    num_prints = num_prints + 1
  end subroutine
end subroutine

program mprog
  use m
  implicit none

  call sub
end program
