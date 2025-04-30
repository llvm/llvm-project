! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!       

module my_mod
!private
type mytype
   integer y
contains
   private
   procedure :: mysub
   procedure, public :: mysub2
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
type(mytype) :: m
print *, 'mysub2 called'
mysub2 = this%mysub
end function mysub2

end module my_mod


program prg
USE CHECK_MOD
use my_mod


type, extends(mytype) :: mytype2
   real u
end type mytype2

type, extends(mytype) :: mytype3
   real t 
end type mytype3

integer results(6)
integer expect(6)
class(mytype2),allocatable :: my2
class(mytype),allocatable :: my
class(mytype3),allocatable :: my3

results = .false.
expect = .true.


allocate(my)
allocate(my2)
allocate(my3)

my%y = 1050
my2%u = 3.5
my3%t = 1.4

results(1) = my%mysub2
results(2) = my2%mysub2()
results(3) = mysub(my3)
results(4) = mysub(my)
results(5) = mysub(my2)
results(6) = my2%mysub2

!print *, all(results .eq. expect)
call check(results,expect,6)

end


