! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!       

module my_mod
!!private

type mytype
   integer y
contains
   private
   procedure :: mysub
end type mytype
contains
integer function mysub(this)
class(mytype) :: this
type(mytype) :: m
print *, 'mysub called'
mysub = extends_type_of(this,m)
end function mysub

end module my_mod

integer function mysub2(this)
use my_mod
class(mytype) :: this
type(mytype) :: m
print *, 'mysub2 called'
mysub2 = extends_type_of(this,m)
end function mysub2



program prg
USE CHECK_MOD
use my_mod

type, extends(mytype) :: mytype2
   real u
contains
procedure :: mysub => mysub2
end type mytype2

type, extends(mytype) :: mytype3
   real t 
end type mytype3

interface
integer function mysub2(this)
import mytype2
class(mytype2) :: this
end function mysub2
end interface

integer results(3)
integer expect(3)
class(mytype2),allocatable :: my2
class(mytype2),allocatable :: my
class(mytype3),allocatable :: my3

results = .false.
expect = .true.


allocate(my)
allocate(my2)
allocate(my3)

my%y = 1050
my2%u = 3.5
my3%t = 1.4

results(1) = my%mysub
results(2) = my%mysub()
results(3) = my2%mysub()

!print *, all(results .eq. expect)
call check(results,expect,3)

end


