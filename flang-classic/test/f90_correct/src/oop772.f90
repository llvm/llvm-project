! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module mod

type t
  integer :: i
  contains
  procedure :: func1
  procedure :: func2
  generic :: func => func1, func2
end type
contains
integer function func1(this, x)
class(t) :: this
integer :: x
func1 = x 
end function

integer function func2(this)
class(t) :: this
func2 = this%i
end function

end module

use mod
logical rslt(1), expect(1)
integer z
type(t) :: o
o%i = -99

z = o%func(o%func())
!print *, z
rslt(1) = z .eq. -99
expect = .true.
call check(rslt,expect,1)

end
