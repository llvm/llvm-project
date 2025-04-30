!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Code based on the reproducer from https://github.com/flang-compiler/flang/issues/1200

module test_implied_do_mod
  implicit none
  private
  public :: test_implied_do
contains
  subroutine test_implied_do()
    implicit none
    integer :: i
    integer :: nfilt
    integer, parameter :: maxfilt = 2
    logical :: better(maxfilt)
    real :: cfilt(maxfilt)
    real :: cstrv
    real :: f
    real :: ffilt(maxfilt)

    ffilt = [2.0, 3.0]
    cfilt = [3.0, 2.0]
    f = 1.0
    cstrv = 4.0

    nfilt = maxfilt
    better(1:nfilt) = [(isbetter([ffilt(i), cfilt(i)], [f, cstrv]), i=1, nfilt)]
    if (.not. any(better(1:nfilt))) then
      print *, 'Right'
    else
      print *, 'Wrong'
      stop 1
    end if
  end subroutine test_implied_do

  function isbetter(fc1, fc2) result(is_better)
    implicit none
    real, intent(in) :: fc1(:)
    real, intent(in) :: fc2(:)
    logical :: is_better

    write (*, *) fc1, fc2
    is_better = all(fc1 < fc2)
  end function isbetter
end module test_implied_do_mod

program test
  use, non_intrinsic :: test_implied_do_mod, only : test_implied_do
  implicit none

  print *, 'Test: Implied do.'
  call test_implied_do()
  print *, 'Succeed: Implied do.'
  print *, "PASS"
end program test
