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

program p
USE CHECK_MOD
use shape_mod


integer results(8)
integer expect(8)
data expect /.true.,.true.,.true.,.false.,.true.,.true.,.true.,.true./
data results /.false.,.false.,.false.,.true.,.false.,.false.,.false.,.false./
type(square),target :: s(10)
type(rectangle),target :: r(5)
type(rectangle) :: r2
type(square) :: s2
class(rectangle),pointer::ptr(:)

ptr => r

results(1) = SAME_TYPE_AS(ptr(1),r(2))
results(2) = SAME_TYPE_AS(r2,ptr(2))
results(3) = SAME_TYPE_AS(ptr(3), s(2)%rectangle)
results(4) = SAME_TYPE_AS(s2, ptr(4))
results(5) = SAME_TYPE_AS(s2%rectangle, ptr(5))

ptr=>s

results(6) = SAME_TYPE_AS(ptr(6),s(1))
results(7) = SAME_TYPE_AS(s2,ptr(7))
results(8) = EXTENDS_TYPE_OF(s(2),ptr(8))

call check(expect,results,8)

end


