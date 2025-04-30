! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

module flang3
use, intrinsic :: iso_c_binding

type, bind(c) :: bar
integer(c_int) :: bar1 = -1
end type bar

type, bind(c) :: zip
type(bar) :: zip1
end type zip

type, bind(c) :: roo
integer(c_int),dimension(99) :: roo1 = 0
type(zip) :: roo2(0:98)
end type roo

type(roo), public, target, dimension(0:98), bind(c) :: foo
save :: foo

end module

SUBROUTINE MOO
USE flang3
DO m=1,2 
write (*,*) foo(m)%roo2
END DO
END SUBROUTINE

print *, "PASS"
end
