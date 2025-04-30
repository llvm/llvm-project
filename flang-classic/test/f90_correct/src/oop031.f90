! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!

module tmod

type my_type
integer :: x(10)
end type my_type

type, extends(my_type) :: my_type2
real :: r
end type my_type2

end module

subroutine sub (x,r,y)
use tmod
class(my_type), allocatable::x
integer r(:)
type(my_type) :: y

r(1) = extends_type_of(x,y)
r(2) = allocated(x)
r(3) = same_type_as(x,y)

end subroutine


program p
USE CHECK_MOD
use tmod

interface
subroutine sub(w,s,z)
use tmod
class(my_type),allocatable::w
integer s(:)
type(my_type) :: z
end subroutine
end interface 

class(my_type2), allocatable::x
type(my_type) :: y
integer results(3)
integer expect(3)
data expect /.true.,.true.,.false./
data results /.false.,.false.,.true./


allocate(x)
call sub(x,results,y)
call check(results,expect,3)

end
