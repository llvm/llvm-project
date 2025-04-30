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
class(shape),pointer::p1
class(rectangle),allocatable::p2
integer results(:)
type(shape) sh
type(rectangle) r
type(square) s
integer st

deallocate(p2)
results(1) = SAME_TYPE_AS(p1,p2)
results(2) = SAME_TYPE_AS(p1,s)
results(3) = SAME_TYPE_AS(p1,r)
results(4) = SAME_TYPE_AS(p1,sh)
results(5) = EXTENDS_TYPE_OF(p1,p2)

end subroutine

program p
USE CHECK_MOD
use shape_mod

interface
subroutine sub2(p1,results,p2)
use shape_mod
class(shape),pointer::p1
class(shape),allocatable::p2
integer results(:)
end subroutine
end interface

integer results(7)
integer expect(7)
type(square),target,allocatable :: s
type(rectangle),target,allocatable :: r
class(shape),pointer::ptr

data expect  /.false.,.true.,.false.,.false.,.true.,.false.,.false./
data results /.true.,.false.,.true.,.true.,.false.,.true.,.true./

allocate(s)
allocate(r)
ptr => s
call sub2(ptr,results,s)
!deallocate(s)
deallocate(r)
results(6) = allocated(s)
results(7) = allocated(r)
call check(expect,results,7)

end


