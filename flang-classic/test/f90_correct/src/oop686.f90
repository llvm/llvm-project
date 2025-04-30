!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
program bug
  type :: otype
    integer member
  end type otype
  type(otype) :: obj
  obj%member = 1
  call subr(obj)
  call check(obj%member, 2, 1)

contains
  subroutine subr(obj)
    class(otype), intent(inout) :: obj
    type(otype) :: new
    new = otype(obj%member + 1)
    select type(obj)
      type is (otype)
        ! pgf90 incorrectly complains here about assignment
        ! to a polymorphic object, despite being in the range
        ! of a type is clause in a select type construct.
        ! (F2008 8.1.5.2 para 5)
        obj = new
    end select
  end subroutine subr
end program bug
