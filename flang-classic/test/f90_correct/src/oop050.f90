! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!       

module shape_mod

type shape
        integer :: color
        logical :: filled
        integer :: x
        integer :: y
end type shape

type, EXTENDS ( shape ) :: rectangle
        integer :: the_length
        integer :: the_width
end type rectangle

type, extends (rectangle) :: square
end type square

end module shape_mod

subroutine sub2(p1,results,p2,i)
use shape_mod
integer i
class(rectangle),pointer::p1(:)
class(shape),pointer::p2(:)
integer results(:)
type(shape) sh
type(rectangle) r
type(square) s

nullify(p1)
results(i) = SAME_TYPE_AS(p1,p2(1))
results(i+1) = SAME_TYPE_AS(p1,s)
results(i+2) = SAME_TYPE_AS(p1,r)
results(i+3) = SAME_TYPE_AS(p1,sh)
results(i+4) = EXTENDS_TYPE_OF(p1,p2(3))

end subroutine

program p
USE CHECK_MOD
use shape_mod

interface
subroutine sub2(p1,results,p2,i)
use shape_mod
class(rectangle),pointer::p1(:)
class(shape),pointer::p2(:)
integer results(:)
end subroutine
end interface

integer results(5)
integer expect(5)
type(square),target :: s(5)
type(rectangle),target :: r(10)
class(rectangle),pointer::ptr(:)
class(shape),pointer::ptr2(:)

data expect  /.true.,.false.,.true.,.false.,.true./
data results /.false.,.true.,.false.,.true.,.false./

ptr2 => r(6:10)
ptr => s
call sub2(ptr,results,ptr2,1)
call check(expect,results,5)

end


