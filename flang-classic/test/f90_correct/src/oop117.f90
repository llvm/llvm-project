! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!       

module my_mod

type,abstract ::  mytype
   integer y
contains
   procedure (mysub1), deferred,nopass :: mysub
   procedure, nopass :: f
end type mytype

interface
integer function mysub1(this) RESULT(R)
import :: mytype
class(mytype) :: this
class(mytype),allocatable :: m
end function
end interface

contains
recursive function f(i) result(r)
 integer i, r

 !print *,i,' called'
 if( i .eq. 1 ) then
   ! base case
   r = 1
 else if( (i/2) * 2 .eq. i )then
   ! divide by two
   r = f(i/2)+1
 else
   ! multiply by three, add one
   r = f(i*3+1)+1
 endif
end function

end module my_mod

integer function mysub1(this) RESULT(R)
use :: my_mod, except => mysub1
class(mytype) :: this
class(mytype),allocatable :: m
R = extends_type_of(this,m)
end function mysub1

program prg
USE CHECK_MOD
use my_mod


type, extends(mytype) :: mytype2
   real u
   contains
   procedure,nopass :: mysub => mysub1
end type mytype2

type, extends(mytype2) :: mytype3
   real t 
end type mytype3

integer results(7)
integer expect(7)
class(mytype2),allocatable :: my2
class(mytype),allocatable :: my
class(mytype3),allocatable :: my3

results = .false.
expect = .true.


allocate(mytype2::my)
allocate(my2)
allocate(my3)

my%y = 1050
my2%u = 3.5
my3%t = 1.4

results(1) = my%mysub(my)
results(2) = my2%mysub(my2)
results(3) = my3%mysub(my3)
results(4) = mysub1(my)
results(5) = mysub1(my2)
results(6) = my2%mysub(my2)
results(7) = my2%f(10) .eq. 7

call check(results,expect,7)

end


