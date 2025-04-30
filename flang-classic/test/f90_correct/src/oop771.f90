! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module mod
  implicit none
  integer z
  type t
  integer val
  contains
    procedure, nopass :: foo
    procedure :: bar
  end type
  contains

  integer function foo()
    foo = z
  end function

  subroutine bar(this,i)
    class(t) :: this
    integer :: i
    this%val = i
  end subroutine
  
end module

use mod
integer x
type(t) :: obj
logical rslts(2), expect(2)

expect = .true.

z = 100
call obj%bar(obj%foo())
!print *, obj%val
rslts(1) = obj%val .eq. 100
z = -99

x = obj%foo()
call obj%bar(x)
!print *, obj%val
rslts(2) = obj%val .eq. -99

call check(rslts, expect, 2)
end


