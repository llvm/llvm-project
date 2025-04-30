!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM
! Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
! Code based on the reproducer from https://github.com/flang-compiler/flang/issues/1146

module public1
  integer, public :: var = 1
  integer, public :: var1 = 11
end module

module public2
  integer, public :: var = 2
end module

module use_public2
  use public2, only : var
  private
end module

program use_1_2
  use public1, only : var1
  use use_public2
  integer :: ret = 0
  ret = foo()
  if (ret /= 1) STOP 1
  print *, "PASS"
contains
  function foo() result(ret)
    use public1, only : var
    integer :: ret
    ret = var
  end function
end program
