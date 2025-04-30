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

subroutine sub2(p1,results,p2)
use shape_mod
class(shape)::p1
class(shape)::p2
integer results(:)
type(rectangle) r
type(square) s

results(1) = SAME_TYPE_AS(p1,s)
results(2) = SAME_TYPE_AS(p2,r)
results(3) = extends_type_of(p1,p2)

end subroutine

program p
USE CHECK_MOD
use shape_mod

interface
subroutine sub2(p1,results,p2)
use shape_mod
class(shape)::p1
class(shape)::p2
integer results(:)
end subroutine
end interface

integer results(3)
integer expect(3)
type(rectangle)::rec
type(square)::s

expect = .true.
result = .false.

call sub2(s,results,rec)
call check(results,expect,3)

end


