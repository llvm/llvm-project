! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!       

module my_mod
type mytype
   integer y
contains
   procedure :: mysub
end type mytype
interface
integer function mysub(this) RESULT(R)
import mytype
class(mytype) :: this
end function mysub
end interface
end module my_mod

integer function mysub(this) RESULT(R)
use :: my_mod, except => mysub
class(mytype) :: this
type(mytype) :: m
R = extends_type_of(this,m)
end function mysub

program prg
USE CHECK_MOD
use my_mod

type, extends(mytype) :: mytype2
   real u
end type mytype2

type, extends(mytype2) :: mytype3
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

results(1) = my%mysub
results(2) = my2%mysub()
results(3) = mysub(my3)
results(4) = mysub(my)
results(5) = mysub(my2)
results(6) = my2%mysub

call check(results,expect,6)

end


