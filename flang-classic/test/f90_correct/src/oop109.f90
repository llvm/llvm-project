! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!       

module my_mod
public
type mytype
   integer y
contains
   procedure,private :: mysub
   procedure :: mysub2
end type mytype
contains
integer function mysub(this)
class(mytype) :: this
type(mytype) :: m
print *, 'mysub called'
mysub = extends_type_of(this,m)
end function mysub

integer function mysub2(this)
class(mytype) :: this
print *, 'mysub2 called'
mysub2 = mysub(this)
end function mysub2

end module my_mod

module my_mod2
use my_mod
public
type,extends(mytype) :: mytype2
   real u
contains
procedure :: mysub => mysub3
end type mytype2
contains
integer function mysub3(this)
class(mytype2) :: this
print *, 'mysub3 called'
mysub3 = mysub2(this)
end function mysub3


end module



program prg
USE CHECK_MOD
use my_mod2

integer results(2)
integer expect(2)

class(mytype2), allocatable :: t
class(mytype), allocatable :: u

expect = .true.
results = .false.

allocate(t)
allocate(u)
results(1) = t%mysub()
results(2) = u%mysub2

call check(results,expect,2)

end


