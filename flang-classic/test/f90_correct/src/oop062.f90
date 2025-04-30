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
class(rectangle),allocatable::p1
type(rectangle),allocatable::p2
integer results(:)
type(rectangle) r
type(shape) s

results(1) = extends_type_of(p1,s)
results(2) = SAME_TYPE_AS(p2,r)
results(3) = extends_type_of(p2,p1)

end subroutine

program p
USE CHECK_MOD
use shape_mod

interface
subroutine sub2(p1,results,p2)
use shape_mod
class(rectangle),allocatable::p1
class(rectangle),allocatable::p2
integer results(:)
end subroutine
end interface

integer results(6)
integer expect(6)
class(square),allocatable :: s
class(rectangle),allocatable :: r
type(rectangle)::rec
type(shape)::sh

expect = .true.
result = .false.

allocate(s)
allocate(rectangle::r)
call sub2(r,results,s)
results(4) = extends_type_of(s,rec)
results(5) = same_type_as(r,rec)
results(6) = extends_type_of(s,r)
call check(results,expect,6)

end


