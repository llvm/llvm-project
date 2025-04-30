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

module test_shape_mod
use shape_mod

type my_type
     type(square) :: sq(10)
     integer :: x
     real :: y
end type my_type

end module test_shape_mod
	

subroutine test_types(s1,s2,results)
use test_shape_mod
class(shape)::s1, s2
integer results(:)
type(my_type) :: z

results(1) = EXTENDS_TYPE_OF(s1,s2)
results(2) = EXTENDS_TYPE_OF(s2,s1)
results(3) = SAME_TYPE_AS(s1,s2)
results(4) = SAME_TYPE_AS(s2,z%sq(1)%rectangle)
results(5) = SAME_TYPE_AS(z%sq(10)%rectangle,s2)
results(6) = SAME_TYPE_AS(s1,z%sq(1))
results(7) = SAME_TYPE_AS(z%sq(10),s1)
end subroutine


program p
USE CHECK_MOD
use shape_mod

interface
subroutine test_types(s1,s2,results)
use shape_mod
class(shape)::s1, s2
integer results(:)
end subroutine
end interface

integer results(7)
integer expect(7)
data expect /.true.,.false.,.false.,.true.,.true.,.true.,.true./
data results /.false.,.true.,.true.,.false.,.false.,.false.,.false./
type(square) :: s
type(rectangle) :: r

call test_types(s,r,results)

call check(expect,results,7)

end


