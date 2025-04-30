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

subroutine test_types(s1,s2,results)
use shape_mod
class(shape)::s1, s2
integer results(:)
class(square), allocatable ::s

allocate(s)
results(1) = EXTENDS_TYPE_OF(s1,s2)
results(2) = EXTENDS_TYPE_OF(s2,s1)
results(3) = SAME_TYPE_AS(s1,s2)
results(4) = SAME_TYPE_AS(s2,s%rectangle)
results(5) = SAME_TYPE_AS(s%rectangle,s2)
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

integer results(5)
integer expect(5)
data expect /.true.,.false.,.false.,.true.,.true./
data results /.false.,.true.,.true.,.false.,.false./
type(square) :: s
type(rectangle) :: r

call test_types(s,r,results)

call check(expect,results,5)

end


