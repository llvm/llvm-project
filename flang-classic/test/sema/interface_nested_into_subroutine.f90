!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

! RUN: %flang -S -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK
!
! CHECK-LABEL: define void @MAIN_()
module base
  implicit none
  integer, public :: a = 10
end module

module intermediate
  use base, only: a
  implicit none
  private
end module

program x
  use base
  implicit none

  interface
    subroutine sub2
      use intermediate
      implicit none
    end subroutine sub2
  end interface

  call sub

contains

subroutine sub
  use intermediate
  implicit none
  print *, a
end subroutine

end program
