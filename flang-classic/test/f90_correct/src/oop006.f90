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
logical l 
integer results(5)
integer expect(5)
data expect /.true.,.false.,.false.,.true.,.true./
data results /.false.,.true.,.true.,.false.,.false./
type(square) :: s
type(rectangle) :: r

results(1) = EXTENDS_TYPE_OF(s,r)
results(2) = EXTENDS_TYPE_OF(r,s)
results(3) = SAME_TYPE_AS(s,r)
results(4) = SAME_TYPE_AS(r,s%rectangle)
results(5) = SAME_TYPE_AS(s%rectangle,r)


call check(expect,results,5)

end


