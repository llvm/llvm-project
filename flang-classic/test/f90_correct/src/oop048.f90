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

subroutine test_types(s1,s2,z,results)
use shape_mod
class(shape)::s1, s2
class(shape)::z(:)
integer results(2)

results(1) = SAME_TYPE_AS(s1,z)
results(2) = SAME_TYPE_AS(z,s2)
end subroutine


program p
USE CHECK_MOD
use shape_mod

interface
subroutine test_types(s1,s2,z,results)
use shape_mod
class(shape)::s1, s2
class(shape)::z(:)
integer results(2)
end subroutine
end interface

type(square)::arr(10)
integer results(2)
integer expect(2)
data expect /.true.,.false./
data results /.false.,.true./
type(square) :: s
type(rectangle) :: r

call test_types(s,r,arr,results)

call check(expect,results,2)

end


